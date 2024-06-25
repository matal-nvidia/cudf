/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "groupby/common/utils.hpp"
#include "groupby/hash/groupby_kernels.cuh"
#include "groupby/hash/row_aggregator.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.cuh>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/static_set.cuh>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <memory>
#include <unordered_set>
#include <utility>

namespace cudf {
namespace groupby {
namespace detail {
namespace hash {
namespace {

// TODO: similar to `contains_table`, using larger CG size like 2 or 4 for nested
// types and `cg_size = 1`for flat data to improve performance
using probing_scheme_type = cuco::linear_probing<
  1,  ///< Number of threads used to handle each input key
  cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                   cudf::nullate::DYNAMIC>>;

template <typename type, std::enable_if_t<std::is_integral_v<type>>* = nullptr>
constexpr __device__ __host__ type divide_round_up(type dividend, type divisor)
{
  assert(divisor != 0);

  return dividend / divisor + (dividend % divisor != 0);
}

__device__ __host__ size_t round_to_multiple_of_8(size_t num)
{
  size_t constexpr multiple_of = 8;

  return divide_round_up(num, multiple_of) * multiple_of;
}

size_t get_previous_multiple_of_8(size_t number) { return number / 8 * 8; }

template <typename SetType>
__device__ cudf::size_type find_local_mapping(cudf::size_type cur_idx,
                                              cudf::size_type num_input_rows,
                                              cudf::size_type* cardinality,
                                              SetType shared_set,
                                              cudf::size_type* local_mapping_index,
                                              cudf::size_type* shared_set_indices,
                                              bitmask_type const* __restrict__ row_bitmask,
                                              bool skip_rows_with_nulls)
{
  cudf::size_type result_idx;
  bool inserted;

  // TODO: ugly
  if (cur_idx < num_input_rows) {
    if (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, cur_idx)) {
      auto const result = shared_set.insert_and_find(cur_idx);
      result_idx        = *result.first;
      inserted          = result.second;

      // inserted a new element
      if (result.second) {
        auto shared_set_index                = atomicAdd(cardinality, 1);
        shared_set_indices[shared_set_index] = cur_idx;
        local_mapping_index[cur_idx]         = shared_set_index;
      }
    }
  }

  // Syncing the thread block is needed so that updates in `local_mapping_index` are visible to all
  // threads in the thread block.
  __syncthreads();

  if (cur_idx < num_input_rows) {
    if (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, cur_idx)) {
      // element was already in set
      if (!inserted) { local_mapping_index[cur_idx] = local_mapping_index[result_idx]; }
    }
  }
}

template <typename SetType>
__device__ cudf::size_type find_global_mapping(cudf::size_type cur_idx,
                                               SetType global_set,
                                               cudf::size_type* shared_set_indices,
                                               cudf::size_type* global_mapping_index,
                                               cudf::size_type shared_set_num_elements)
{
  auto input_idx = shared_set_indices[cur_idx];
  auto result    = global_set.insert_and_find(input_idx);
  global_mapping_index[blockIdx.x * shared_set_num_elements + cur_idx] = *result.first;
}

/*
 * Inserts keys into the shared memory hash set, and stores the row index of the local
 * pre-aggregate table in `local_mapping_index`. If the number of unique keys found in a
 * threadblock exceeds `cardinality_threshold`, the threads in that block will exit without updating
 * `global_set` or setting `global_mapping_index`. Else, we insert the unique keys found to the
 * global hash set, and save the row index of the global sparse table in `global_mapping_index`.
 */
template <class SetRef,
          cudf::size_type shared_set_num_elements,
          cudf::size_type cardinality_threshold,
          typename GlobalSetType,
          typename KeyEqual,
          typename RowHasher,
          class WindowExtent>
__global__ void compute_mapping_indices(GlobalSetType global_set,
                                        cudf::size_type num_input_rows,
                                        WindowExtent window_extent,
                                        cuco::empty_key<cudf::size_type> empty_key_sentinel,
                                        KeyEqual d_key_equal,
                                        RowHasher d_row_hash,
                                        cudf::size_type* local_mapping_index,
                                        cudf::size_type* global_mapping_index,
                                        cudf::size_type* block_cardinality,
                                        bool* direct_aggregations,
                                        bitmask_type const* __restrict__ row_bitmask,
                                        bool skip_rows_with_nulls)
{
  __shared__ cudf::size_type shared_set_indices[shared_set_num_elements];

  // Shared set initialization
  __shared__ typename SetRef::window_type windows[window_extent.value()];
  auto storage = SetRef::storage_ref_type(window_extent, windows);
  auto shared_set =
    SetRef(empty_key_sentinel, d_key_equal, probing_scheme_type{d_row_hash}, {}, storage);
  auto const block = cooperative_groups::this_thread_block();
  shared_set.initialize(block);
  block.sync();

  auto shared_insert_ref = std::move(shared_set).with(cuco::insert_and_find);

  __shared__ cudf::size_type cardinality;

  if (threadIdx.x == 0) { cardinality = 0; }

  __syncthreads();

  int num_loops = divide_round_up(num_input_rows, (cudf::size_type)(blockDim.x * gridDim.x));
  auto end_idx  = num_loops * blockDim.x * gridDim.x;

  for (auto cur_idx = blockDim.x * blockIdx.x + threadIdx.x; cur_idx < end_idx;
       cur_idx += blockDim.x * gridDim.x) {
    find_local_mapping(cur_idx,
                       num_input_rows,
                       &cardinality,
                       shared_insert_ref,
                       local_mapping_index,
                       shared_set_indices,
                       row_bitmask,
                       skip_rows_with_nulls);
    __syncthreads();

    if (cardinality >= cardinality_threshold) {
      if (threadIdx.x == 0) { *direct_aggregations = true; }
      break;
    }

    __syncthreads();
  }

  // Insert unique keys from shared to global hash set
  if (cardinality < cardinality_threshold) {
    for (auto cur_idx = threadIdx.x; cur_idx < cardinality; cur_idx += blockDim.x) {
      find_global_mapping(
        cur_idx, global_set, shared_set_indices, global_mapping_index, shared_set_num_elements);
    }
  }

  if (threadIdx.x == 0) block_cardinality[blockIdx.x] = cardinality;
}

__device__ void calculate_columns_to_aggregate(int& col_start,
                                               int& col_end,
                                               cudf::mutable_table_device_view output_values,
                                               int num_input_cols,
                                               std::byte** s_aggregates_pointer,
                                               bool** s_aggregates_valid_pointer,
                                               std::byte* shared_set_aggregates,
                                               cudf::size_type cardinality,
                                               int total_agg_size)
{
  if (threadIdx.x == 0) {
    col_start           = col_end;
    int bytes_allocated = 0;
    int valid_col_size  = round_to_multiple_of_8(sizeof(bool) * cardinality);
    while ((bytes_allocated < total_agg_size) && (col_end < num_input_cols)) {
      int next_col_size =
        round_to_multiple_of_8(sizeof(output_values.column(col_end).type()) * cardinality);
      int next_col_total_size = valid_col_size + next_col_size;

      if (bytes_allocated + next_col_total_size > total_agg_size) { break; }

      s_aggregates_pointer[col_end] = shared_set_aggregates + bytes_allocated;
      s_aggregates_valid_pointer[col_end] =
        reinterpret_cast<bool*>(shared_set_aggregates + bytes_allocated + next_col_size);

      bytes_allocated += next_col_total_size;
      col_end++;
    }
  }
}

__device__ void initialize_shared_memory_aggregates(int col_start,
                                                    int col_end,
                                                    cudf::mutable_table_device_view output_values,
                                                    std::byte** s_aggregates_pointer,
                                                    bool** s_aggregates_valid_pointer,
                                                    cudf::size_type cardinality,
                                                    cudf::aggregation::Kind const* aggs)
{
  for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
    for (auto idx = threadIdx.x; idx < cardinality; idx += blockDim.x) {
      cudf::detail::dispatch_type_and_aggregation(output_values.column(col_idx).type(),
                                                  aggs[col_idx],
                                                  gqe::initialize_shmem{},
                                                  s_aggregates_pointer[col_idx],
                                                  idx,
                                                  s_aggregates_valid_pointer[col_idx]);
    }
  }
}

__device__ void compute_pre_aggregrates(int col_start,
                                        int col_end,
                                        cudf::table_device_view input_values,
                                        cudf::size_type num_input_rows,
                                        cudf::size_type* local_mapping_index,
                                        std::byte** s_aggregates_pointer,
                                        bool** s_aggregates_valid_pointer,
                                        cudf::aggregation::Kind const* aggs,
                                        bitmask_type const* row_bitmask,
                                        bool skip_rows_with_nulls)
{
  for (auto cur_idx = blockDim.x * blockIdx.x + threadIdx.x; cur_idx < num_input_rows;
       cur_idx += blockDim.x * gridDim.x) {
    // TODO: ugly
    if (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, cur_idx)) {
      auto map_idx = local_mapping_index[cur_idx];

      for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
        auto input_col = input_values.column(col_idx);

        cudf::detail::dispatch_type_and_aggregation(input_col.type(),
                                                    aggs[col_idx],
                                                    gqe::shmem_element_aggregator{},
                                                    s_aggregates_pointer[col_idx],
                                                    map_idx,
                                                    s_aggregates_valid_pointer[col_idx],
                                                    input_col,
                                                    cur_idx);
      }
    }
  }
}

template <int shared_set_num_elements>
__device__ void compute_final_aggregates(int col_start,
                                         int col_end,
                                         cudf::table_device_view input_values,
                                         cudf::mutable_table_device_view output_values,
                                         cudf::size_type cardinality,
                                         cudf::size_type* global_mapping_index,
                                         std::byte** s_aggregates_pointer,
                                         bool** s_aggregates_valid_pointer,
                                         cudf::aggregation::Kind const* aggs)
{
  for (auto cur_idx = threadIdx.x; cur_idx < cardinality; cur_idx += blockDim.x) {
    auto out_idx = global_mapping_index[blockIdx.x * shared_set_num_elements + cur_idx];
    for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
      auto output_col = output_values.column(col_idx);

      cudf::detail::dispatch_type_and_aggregation(input_values.column(col_idx).type(),
                                                  aggs[col_idx],
                                                  gqe::gmem_element_aggregator{},
                                                  output_col,
                                                  out_idx,
                                                  input_values.column(col_idx),
                                                  s_aggregates_pointer[col_idx],
                                                  cur_idx,
                                                  s_aggregates_valid_pointer[col_idx]);
    }
  }
}

/* Takes the local_mapping_index and global_mapping_index to compute
 * pre (shared) and final (global) aggregates*/
template <cudf::size_type shared_set_num_elements, cudf::size_type cardinality_threshold>
__global__ void compute_aggregates(cudf::size_type* local_mapping_index,
                                   cudf::size_type* global_mapping_index,
                                   cudf::size_type* block_cardinality,
                                   cudf::table_device_view input_values,
                                   cudf::mutable_table_device_view output_values,
                                   cudf::size_type num_input_rows,
                                   cudf::aggregation::Kind const* aggs,
                                   int total_agg_size,
                                   int pointer_size,
                                   bitmask_type const* row_bitmask,
                                   bool skip_rows_with_null)
{
  cudf::size_type cardinality = block_cardinality[blockIdx.x];
  if (cardinality >= cardinality_threshold) { return; }

  int num_input_cols = output_values.num_columns();
  extern __shared__ std::byte shared_set_aggregates[];
  std::byte** s_aggregates_pointer =
    reinterpret_cast<std::byte**>(shared_set_aggregates + total_agg_size);
  bool** s_aggregates_valid_pointer =
    reinterpret_cast<bool**>(shared_set_aggregates + total_agg_size + pointer_size);

  __shared__ int col_start;
  __shared__ int col_end;
  if (threadIdx.x == 0) {
    col_start = 0;
    col_end   = 0;
  }
  __syncthreads();

  while (col_end < num_input_cols) {
    calculate_columns_to_aggregate(col_start,
                                   col_end,
                                   output_values,
                                   num_input_cols,
                                   s_aggregates_pointer,
                                   s_aggregates_valid_pointer,
                                   shared_set_aggregates,
                                   cardinality,
                                   total_agg_size);
    __syncthreads();

    initialize_shared_memory_aggregates(col_start,
                                        col_end,
                                        output_values,
                                        s_aggregates_pointer,
                                        s_aggregates_valid_pointer,
                                        cardinality,
                                        aggs);
    __syncthreads();

    compute_pre_aggregrates(col_start,
                            col_end,
                            input_values,
                            num_input_rows,
                            local_mapping_index,
                            s_aggregates_pointer,
                            s_aggregates_valid_pointer,
                            aggs,
                            row_bitmask,
                            skip_rows_with_null);
    __syncthreads();

    compute_final_aggregates<shared_set_num_elements>(col_start,
                                                      col_end,
                                                      input_values,
                                                      output_values,
                                                      cardinality,
                                                      global_mapping_index,
                                                      s_aggregates_pointer,
                                                      s_aggregates_valid_pointer,
                                                      aggs);
    __syncthreads();
  }
}

template <typename SetType>
struct compute_direct_aggregates {
  SetType set;
  cudf::table_device_view input_values;
  cudf::mutable_table_device_view output_values;
  cudf::aggregation::Kind const* __restrict__ aggs;
  bitmask_type const* __restrict__ row_bitmask;
  bool skip_rows_with_nulls;
  cudf::size_type* block_cardinality;
  int stride;
  int block_size;
  cudf::size_type cardinality_threshold;

  compute_direct_aggregates(SetType set,
                            cudf::table_device_view input_values,
                            cudf::mutable_table_device_view output_values,
                            cudf::aggregation::Kind const* aggs,
                            bitmask_type const* row_bitmask,
                            bool skip_rows_with_nulls,
                            cudf::size_type* block_cardinality,
                            int stride,
                            int block_size,
                            cudf::size_type cardinality_threshold)
    : set(set),
      input_values(input_values),
      output_values(output_values),
      aggs(aggs),
      row_bitmask(row_bitmask),
      skip_rows_with_nulls(skip_rows_with_nulls),
      block_cardinality(block_cardinality),
      stride(stride),
      block_size(block_size),
      cardinality_threshold(cardinality_threshold)
  {
  }

  __device__ void operator()(cudf::size_type i)
  {
    if (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, i)) {
      int block_id = (i % stride) / block_size;
      if (block_cardinality[block_id] >= cardinality_threshold) {
        auto const result = set.insert_and_find(i);
        cudf::detail::aggregate_row<true, true>(
          output_values, *result.first, input_values, i, aggs);
      }
    }
  }
};

// TODO: see diff
template <typename SetType>
void extract_populated_keys(SetType const& key_set,
                            rmm::device_uvector<cudf::size_type>& populated_keys,
                            rmm::cuda_stream_view stream)
{
  auto const keys_end = key_set.retrieve_all(populated_keys.begin(), stream.value());

  populated_keys.resize(std::distance(populated_keys.begin(), keys_end), stream);
}

// struct initialize_sparse_table {
//   cudf::size_type const* row_indices;
//   cudf::mutable_table_device_view sparse_table;
//   cudf::aggregation::Kind const* __restrict__ aggs;

//   initialize_sparse_table(cudf::size_type const* row_indices,
//                           cudf::mutable_table_device_view sparse_table,
//                           cudf::aggregation::Kind const* aggs)
//     : row_indices(row_indices), sparse_table(sparse_table), aggs(aggs)
//   {
//   }

//   __device__ void operator()(cudf::size_type i)
//   {
//     auto key_idx = row_indices[i];
//     for (auto col_idx = 0; col_idx < sparse_table.num_columns(); col_idx++) {
//       cudf::detail::dispatch_type_and_aggregation(sparse_table.column(col_idx).type(),
//                                                   aggs[col_idx],
//                                                   gqe::initialize_gmem{},
//                                                   sparse_table.column(col_idx),
//                                                   key_idx);
//     }
//   }
// };

// TODO: see diff

// template <typename GlobalSetType>
// auto create_sparse_results_table(cudf::table_view const& flattened_values,
//                                  const cudf::aggregation::Kind* d_aggs,
//                                  std::vector<cudf::aggregation::Kind> aggs,
//                                  bool direct_aggregations,
//                                  GlobalSetType const& global_set,
//                                  rmm::device_uvector<cudf::size_type>& populated_keys,
//                                  rmm::cuda_stream_view stream)
// {
//   // TODO single allocation - room for performance improvement
//   std::vector<std::unique_ptr<cudf::column>> sparse_columns;
//   std::transform(
//     flattened_values.begin(),
//     flattened_values.end(),
//     aggs.begin(),
//     std::back_inserter(sparse_columns),
//     [stream](auto const& col, auto const& agg) {
//       bool nullable = (agg == cudf::aggregation::COUNT_VALID or agg ==
//       cudf::aggregation::COUNT_ALL)
//                         ? false
//                         : (col.has_nulls() or agg == cudf::aggregation::VARIANCE or
//                            agg == cudf::aggregation::STD);
//       auto mask_flag = (nullable) ? cudf::mask_state::ALL_NULL : cudf::mask_state::UNALLOCATED;

//       auto col_type = cudf::is_dictionary(col.type())
//                         ? cudf::dictionary_column_view(col).keys().type()
//                         : col.type();

//       auto target_type = cudf::detail::target_type(col_type, agg);
//       return make_fixed_width_column(
//         cudf::detail::target_type(col_type, agg), col.size(), mask_flag, stream);
//     });

//   cudf::table sparse_table(std::move(sparse_columns));

//   // If no direct aggregations, initialize the sparse table
//   // only for the keys inserted in global hash set
//   if (!direct_aggregations) {
//     auto d_sparse_table = cudf::mutable_table_device_view::create(sparse_table, stream);
//     extract_populated_keys(global_set, populated_keys, stream);
//     thrust::for_each_n(rmm::exec_policy(stream),
//                        thrust::make_counting_iterator(0),
//                        populated_keys.size(),
//                        initialize_sparse_table{populated_keys.data(), *d_sparse_table,
//                        d_aggs});
//   }

//   // Else initialise the whole table
//   else {
//     cudf::mutable_table_view sparse_table_view = sparse_table.mutable_view();
//     cudf::detail::initialize_with_identity(sparse_table_view, aggs, stream);
//   }

//   return sparse_table;
// }

int find_num_sms()
{
  int dev_id{-1};
  CUDF_CUDA_TRY(cudaGetDevice(&dev_id));

  int num_sms{-1};
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  return num_sms;
}

template <typename FuncType>
int find_grid_size(FuncType func, int block_size, cudf::size_type num_input_rows, int num_sms)
{
  int max_active_blocks{-1};
  CUDF_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, func, block_size, 0));

  auto max_grid_size       = max_active_blocks * num_sms;
  int needed_active_blocks = divide_round_up(num_input_rows, block_size);
  return std::min(max_grid_size, needed_active_blocks);
}

template <typename FuncType>
size_t find_shmem_size(FuncType func, int block_size, int grid_size, int num_sms)
{
  auto active_blocks_per_sm = divide_round_up(grid_size, num_sms);

  size_t dynamic_smem_size;
  CUDF_CUDA_TRY(cudaOccupancyAvailableDynamicSMemPerBlock(
    &dynamic_smem_size, func, active_blocks_per_sm, block_size));
  return get_previous_multiple_of_8(0.5 * dynamic_smem_size);
}

template <cudf::size_type shared_set_num_elements, cudf::size_type cardinality_threshold>
void launch_compute_aggregates(int block_size,
                               int grid_size,
                               int num_sms,
                               cudf::size_type* local_mapping_index,
                               cudf::size_type* global_mapping_index,
                               cudf::size_type* block_cardinality,
                               cudf::table_device_view input_values,
                               cudf::mutable_table_device_view output_values,
                               cudf::size_type num_input_rows,
                               cudf::aggregation::Kind const* aggs,
                               rmm::cuda_stream_view stream,
                               bitmask_type const* __restrict__ row_bitmask,
                               bool skip_rows_with_nulls)
{
  auto compute_aggregates_fn_ptr =
    compute_aggregates<shared_set_num_elements, cardinality_threshold>;
  size_t d_shmem_size = find_shmem_size(compute_aggregates_fn_ptr, block_size, grid_size, num_sms);

  // For each aggregation, need two pointers to arrays in shmem
  // One where the aggregation is performed, one indicating the validity of the aggregation
  auto shmem_agg_pointer_size =
    round_to_multiple_of_8(sizeof(std::byte*) * output_values.num_columns());
  // The rest of shmem is utilized for the actual arrays in shmem
  auto shmem_agg_size = d_shmem_size - shmem_agg_pointer_size * 2;

  compute_aggregates<shared_set_num_elements, cardinality_threshold>
    <<<grid_size, block_size, d_shmem_size, stream>>>(local_mapping_index,
                                                      global_mapping_index,
                                                      block_cardinality,
                                                      input_values,
                                                      output_values,
                                                      num_input_rows,
                                                      aggs,
                                                      shmem_agg_size,
                                                      shmem_agg_pointer_size,
                                                      row_bitmask,
                                                      skip_rows_with_nulls);
}

/**
 * @brief List of aggregation operations that can be computed with a hash-based
 * implementation.
 */
constexpr std::array<aggregation::Kind, 12> hash_aggregations{aggregation::SUM,
                                                              aggregation::PRODUCT,
                                                              aggregation::MIN,
                                                              aggregation::MAX,
                                                              aggregation::COUNT_VALID,
                                                              aggregation::COUNT_ALL,
                                                              aggregation::ARGMIN,
                                                              aggregation::ARGMAX,
                                                              aggregation::SUM_OF_SQUARES,
                                                              aggregation::MEAN,
                                                              aggregation::STD,
                                                              aggregation::VARIANCE};

// Could be hash: SUM, PRODUCT, MIN, MAX, COUNT_VALID, COUNT_ALL, ANY, ALL,
// Compound: MEAN(SUM, COUNT_VALID), VARIANCE, STD(MEAN (SUM, COUNT_VALID), COUNT_VALID),
// ARGMAX, ARGMIN

// TODO replace with std::find in C++20 onwards.
template <class T, size_t N>
constexpr bool array_contains(std::array<T, N> const& haystack, T needle)
{
  for (auto const& val : haystack) {
    if (val == needle) return true;
  }
  return false;
}

/**
 * @brief Indicates whether the specified aggregation operation can be computed
 * with a hash-based implementation.
 *
 * @param t The aggregation operation to verify
 * @return true `t` is valid for a hash based groupby
 * @return false `t` is invalid for a hash based groupby
 */
bool constexpr is_hash_aggregation(aggregation::Kind t)
{
  return array_contains(hash_aggregations, t);
}

class groupby_simple_aggregations_collector final
  : public cudf::detail::simple_aggregations_collector {
 public:
  using cudf::detail::simple_aggregations_collector::visit;

  std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                  cudf::detail::min_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(col_type.id() == type_id::STRING ? make_argmin_aggregation()
                                                    : make_min_aggregation());
    return aggs;
  }

  std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                  cudf::detail::max_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(col_type.id() == type_id::STRING ? make_argmax_aggregation()
                                                    : make_max_aggregation());
    return aggs;
  }

  std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                  cudf::detail::mean_aggregation const&) override
  {
    (void)col_type;
    CUDF_EXPECTS(is_fixed_width(col_type), "MEAN aggregation expects fixed width type");
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(make_sum_aggregation());
    // COUNT_VALID
    aggs.push_back(make_count_aggregation());

    return aggs;
  }

  std::vector<std::unique_ptr<aggregation>> visit(data_type,
                                                  cudf::detail::var_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(make_sum_aggregation());
    // COUNT_VALID
    aggs.push_back(make_count_aggregation());

    return aggs;
  }

  std::vector<std::unique_ptr<aggregation>> visit(data_type,
                                                  cudf::detail::std_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(make_sum_aggregation());
    // COUNT_VALID
    aggs.push_back(make_count_aggregation());

    return aggs;
  }

  std::vector<std::unique_ptr<aggregation>> visit(
    data_type, cudf::detail::correlation_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(make_sum_aggregation());
    // COUNT_VALID
    aggs.push_back(make_count_aggregation());

    return aggs;
  }
};

template <typename SetType>
class hash_compound_agg_finalizer final : public cudf::detail::aggregation_finalizer {
  column_view col;
  data_type result_type;
  cudf::detail::result_cache* sparse_results;
  cudf::detail::result_cache* dense_results;
  device_span<size_type const> gather_map;
  SetType set;
  bitmask_type const* __restrict__ row_bitmask;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

 public:
  using cudf::detail::aggregation_finalizer::visit;

  hash_compound_agg_finalizer(column_view col,
                              cudf::detail::result_cache* sparse_results,
                              cudf::detail::result_cache* dense_results,
                              device_span<size_type const> gather_map,
                              SetType set,
                              bitmask_type const* row_bitmask,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
    : col(col),
      sparse_results(sparse_results),
      dense_results(dense_results),
      gather_map(gather_map),
      set(set),
      row_bitmask(row_bitmask),
      stream(stream),
      mr(mr)
  {
    result_type = cudf::is_dictionary(col.type()) ? cudf::dictionary_column_view(col).keys().type()
                                                  : col.type();
  }

  auto to_dense_agg_result(cudf::aggregation const& agg)
  {
    auto s                  = sparse_results->get_result(col, agg);
    auto dense_result_table = cudf::detail::gather(table_view({std::move(s)}),
                                                   gather_map,
                                                   out_of_bounds_policy::DONT_CHECK,
                                                   cudf::detail::negative_index_policy::NOT_ALLOWED,
                                                   stream,
                                                   mr);
    return std::move(dense_result_table->release()[0]);
  }

  // Enables conversion of ARGMIN/ARGMAX into MIN/MAX
  auto gather_argminmax(aggregation const& agg)
  {
    auto arg_result = to_dense_agg_result(agg);
    // We make a view of ARG(MIN/MAX) result without a null mask and gather
    // using this map. The values in data buffer of ARG(MIN/MAX) result
    // corresponding to null values was initialized to ARG(MIN/MAX)_SENTINEL
    // which is an out of bounds index value (-1) and causes the gathered
    // value to be null.
    column_view null_removed_map(
      data_type(type_to_id<size_type>()),
      arg_result->size(),
      static_cast<void const*>(arg_result->view().template data<size_type>()),
      nullptr,
      0);
    auto gather_argminmax =
      cudf::detail::gather(table_view({col}),
                           null_removed_map,
                           arg_result->nullable() ? cudf::out_of_bounds_policy::NULLIFY
                                                  : cudf::out_of_bounds_policy::DONT_CHECK,
                           cudf::detail::negative_index_policy::NOT_ALLOWED,
                           stream,
                           mr);
    return std::move(gather_argminmax->release()[0]);
  }

  // Declare overloads for each kind of aggregation to dispatch
  void visit(cudf::aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;
    dense_results->add_result(col, agg, to_dense_agg_result(agg));
  }

  void visit(cudf::detail::min_aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;
    if (result_type.id() == type_id::STRING) {
      auto transformed_agg = make_argmin_aggregation();
      dense_results->add_result(col, agg, gather_argminmax(*transformed_agg));
    } else {
      dense_results->add_result(col, agg, to_dense_agg_result(agg));
    }
  }

  void visit(cudf::detail::max_aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;

    if (result_type.id() == type_id::STRING) {
      auto transformed_agg = make_argmax_aggregation();
      dense_results->add_result(col, agg, gather_argminmax(*transformed_agg));
    } else {
      dense_results->add_result(col, agg, to_dense_agg_result(agg));
    }
  }

  void visit(cudf::detail::mean_aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;

    auto sum_agg   = make_sum_aggregation();
    auto count_agg = make_count_aggregation();
    this->visit(*sum_agg);
    this->visit(*count_agg);
    column_view sum_result   = dense_results->get_result(col, *sum_agg);
    column_view count_result = dense_results->get_result(col, *count_agg);

    auto result =
      cudf::detail::binary_operation(sum_result,
                                     count_result,
                                     binary_operator::DIV,
                                     cudf::detail::target_type(result_type, aggregation::MEAN),
                                     stream,
                                     mr);
    dense_results->add_result(col, agg, std::move(result));
  }

  void visit(cudf::detail::var_aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;

    auto sum_agg   = make_sum_aggregation();
    auto count_agg = make_count_aggregation();
    this->visit(*sum_agg);
    this->visit(*count_agg);
    column_view sum_result   = sparse_results->get_result(col, *sum_agg);
    column_view count_result = sparse_results->get_result(col, *count_agg);

    auto values_view = column_device_view::create(col, stream);
    auto sum_view    = column_device_view::create(sum_result, stream);
    auto count_view  = column_device_view::create(count_result, stream);

    auto var_result = make_fixed_width_column(
      cudf::detail::target_type(result_type, agg.kind), col.size(), mask_state::ALL_NULL, stream);
    auto var_result_view = mutable_column_device_view::create(var_result->mutable_view(), stream);
    mutable_table_view var_table_view{{var_result->mutable_view()}};
    cudf::detail::initialize_with_identity(var_table_view, {agg.kind}, stream);

    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      col.size(),
      ::cudf::detail::var_hash_functor{
        set, row_bitmask, *var_result_view, *values_view, *sum_view, *count_view, agg._ddof});
    sparse_results->add_result(col, agg, std::move(var_result));
    dense_results->add_result(col, agg, to_dense_agg_result(agg));
  }

  void visit(cudf::detail::std_aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;
    auto var_agg = make_variance_aggregation(agg._ddof);
    this->visit(*dynamic_cast<cudf::detail::var_aggregation*>(var_agg.get()));
    column_view variance = dense_results->get_result(col, *var_agg);

    auto result = cudf::detail::unary_operation(variance, unary_operator::SQRT, stream, mr);
    dense_results->add_result(col, agg, std::move(result));
  }
};
// flatten aggs to filter in single pass aggs
std::tuple<table_view, std::vector<aggregation::Kind>, std::vector<std::unique_ptr<aggregation>>>
flatten_single_pass_aggs(host_span<aggregation_request const> requests)
{
  std::vector<column_view> columns;
  std::vector<std::unique_ptr<aggregation>> aggs;
  std::vector<aggregation::Kind> agg_kinds;

  for (auto const& request : requests) {
    auto const& agg_v = request.aggregations;

    std::unordered_set<aggregation::Kind> agg_kinds_set;
    auto insert_agg = [&](column_view const& request_values, std::unique_ptr<aggregation>&& agg) {
      if (agg_kinds_set.insert(agg->kind).second) {
        agg_kinds.push_back(agg->kind);
        aggs.push_back(std::move(agg));
        columns.push_back(request_values);
      }
    };

    auto values_type = cudf::is_dictionary(request.values.type())
                         ? cudf::dictionary_column_view(request.values).keys().type()
                         : request.values.type();
    for (auto&& agg : agg_v) {
      groupby_simple_aggregations_collector collector;

      for (auto& agg_s : agg->get_simple_aggregations(values_type, collector)) {
        insert_agg(request.values, std::move(agg_s));
      }
    }
  }

  return std::make_tuple(table_view(columns), std::move(agg_kinds), std::move(aggs));
}

/**
 * @brief Gather sparse results into dense using `gather_map` and add to
 * `dense_cache`
 *
 * @see groupby_null_templated()
 */
template <typename SetType>
void sparse_to_dense_results(table_view const& keys,
                             host_span<aggregation_request const> requests,
                             cudf::detail::result_cache* sparse_results,
                             cudf::detail::result_cache* dense_results,
                             device_span<size_type const> gather_map,
                             SetType set,
                             bool keys_have_nulls,
                             null_policy include_null_keys,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto row_bitmask =
    cudf::detail::bitmask_and(keys, stream, rmm::mr::get_current_device_resource()).first;
  bool skip_key_rows_with_nulls = keys_have_nulls and include_null_keys == null_policy::EXCLUDE;
  bitmask_type const* row_bitmask_ptr =
    skip_key_rows_with_nulls ? static_cast<bitmask_type*>(row_bitmask.data()) : nullptr;

  for (auto const& request : requests) {
    auto const& agg_v = request.aggregations;
    auto const& col   = request.values;

    // Given an aggregation, this will get the result from sparse_results and
    // convert and return dense, compacted result
    auto finalizer = hash_compound_agg_finalizer(
      col, sparse_results, dense_results, gather_map, set, row_bitmask_ptr, stream, mr);
    for (auto&& agg : agg_v) {
      agg->finalize(finalizer);
    }
  }
}

// make table that will hold sparse results
auto create_sparse_results_table(table_view const& flattened_values,
                                 std::vector<aggregation::Kind> aggs,
                                 rmm::cuda_stream_view stream)
{
  // TODO single allocation - room for performance improvement
  std::vector<std::unique_ptr<column>> sparse_columns;
  std::transform(
    flattened_values.begin(),
    flattened_values.end(),
    aggs.begin(),
    std::back_inserter(sparse_columns),
    [stream](auto const& col, auto const& agg) {
      bool nullable =
        (agg == aggregation::COUNT_VALID or agg == aggregation::COUNT_ALL)
          ? false
          : (col.has_nulls() or agg == aggregation::VARIANCE or agg == aggregation::STD);
      auto mask_flag = (nullable) ? mask_state::ALL_NULL : mask_state::UNALLOCATED;

      auto col_type = cudf::is_dictionary(col.type())
                        ? cudf::dictionary_column_view(col).keys().type()
                        : col.type();

      return make_fixed_width_column(
        cudf::detail::target_type(col_type, agg), col.size(), mask_flag, stream);
    });

  table sparse_table(std::move(sparse_columns));
  mutable_table_view table_view = sparse_table.mutable_view();
  cudf::detail::initialize_with_identity(table_view, aggs, stream);
  return sparse_table;
}

// TODO: what about multipass
/**
 * @brief Computes all aggregations from `requests` that require a single pass
 * over the data and stores the results in `sparse_results`
 */
template <typename SetType>
void compute_single_pass_aggs(table_view const& keys,
                              host_span<aggregation_request const> requests,
                              cudf::detail::result_cache* sparse_results,
                              SetType set,
                              bool keys_have_nulls,
                              null_policy include_null_keys,
                              rmm::cuda_stream_view stream)
{
  // flatten the aggs to a table that can be operated on by aggregate_row
  auto const [flattened_values, agg_kinds, aggs] = flatten_single_pass_aggs(requests);

  // make table that will hold sparse results
  table sparse_table = create_sparse_results_table(flattened_values, agg_kinds, stream);
  // prepare to launch kernel to do the actual aggregation
  auto d_sparse_table = mutable_table_device_view::create(sparse_table, stream);
  auto d_values       = table_device_view::create(flattened_values, stream);
  auto const d_aggs   = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, rmm::mr::get_current_device_resource());
  auto const skip_key_rows_with_nulls =
    keys_have_nulls and include_null_keys == null_policy::EXCLUDE;

  auto row_bitmask =
    skip_key_rows_with_nulls
      ? cudf::detail::bitmask_and(keys, stream, rmm::mr::get_current_device_resource()).first
      : rmm::device_buffer{};

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    keys.num_rows(),
    hash::compute_single_pass_aggs_fn{set,
                                      *d_values,
                                      *d_sparse_table,
                                      d_aggs.data(),
                                      static_cast<bitmask_type*>(row_bitmask.data()),
                                      skip_key_rows_with_nulls});
  // Add results back to sparse_results cache
  auto sparse_result_cols = sparse_table.release();
  for (size_t i = 0; i < aggs.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    sparse_results->add_result(
      flattened_values.column(i), *aggs[i], std::move(sparse_result_cols[i]));
  }
}

/**
 * @brief Computes and returns a device vector containing all populated keys in
 * `map`.
 */
template <typename SetType>
rmm::device_uvector<size_type> extract_populated_keys(SetType const& key_set,
                                                      size_type num_keys,
                                                      rmm::cuda_stream_view stream)
{
  rmm::device_uvector<size_type> populated_keys(num_keys, stream);
  auto const keys_end = key_set.retrieve_all(populated_keys.begin(), stream.value());

  populated_keys.resize(std::distance(populated_keys.begin(), keys_end), stream);
  return populated_keys;
}

template <typename SetType, typename KeyEqual, typename RowHasher>
void compute_single_pass_set_aggs(
  cudf::table_view const& keys,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  SetType& global_set,
  bool keys_have_nulls,
  null_policy include_null_keys,
  rmm::cuda_stream_view stream,
  cuco::empty_key<cudf::size_type> empty_key_sentinel,
  KeyEqual d_key_equal,
  RowHasher d_row_hash)
{
  // TODO: change flatten function
  auto const [flattened_values, agg_kinds, aggs] = flatten_single_pass_aggs(requests);

  auto const num_input_rows = keys.num_rows();
  auto d_values             = cudf::table_device_view::create(flattened_values, stream);
  auto const d_aggs         = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, rmm::mr::get_current_device_resource());
  auto const skip_key_rows_with_nulls =
    keys_have_nulls and include_null_keys == null_policy::EXCLUDE;

  auto row_bitmask =
    skip_key_rows_with_nulls
      ? cudf::detail::bitmask_and(keys, stream, rmm::mr::get_current_device_resource()).first
      : rmm::device_buffer{};

  auto constexpr window_size                      = 1;
  constexpr int block_size                        = 128;
  constexpr cudf::size_type cardinality_threshold = 128;

  // We add additional `block_size`, because after the number of elements in the local hash set
  // exceeds the threshold, all threads in the thread block can still insert one more element.
  constexpr cudf::size_type shared_set_num_elements = cardinality_threshold + block_size;
  constexpr float load_factor                       = 0.7;
  constexpr std::size_t shared_set_size             = (1.0 / load_factor) * shared_set_num_elements;

  using extent_type            = cuco::extent<cudf::size_type, shared_set_size>;
  using shared_set_type        = cuco::static_set<cudf::size_type,
                                           extent_type,
                                           cuda::thread_scope_block,
                                           decltype(d_key_equal),
                                           probing_scheme_type,
                                           cuco::cuda_allocator<cudf::size_type>,
                                           cuco::storage<window_size>>;
  using shared_set_ref_type    = typename shared_set_type::ref_type<>;
  auto constexpr window_extent = cuco::make_window_extent<shared_set_ref_type>(extent_type{});

  auto global_set_ref = global_set.ref(cuco::op::insert_and_find);

  int num_sms                         = find_num_sms();
  auto compute_mapping_indices_fn_ptr = compute_mapping_indices<shared_set_ref_type,
                                                                shared_set_num_elements,
                                                                cardinality_threshold,
                                                                decltype(global_set_ref),
                                                                KeyEqual,
                                                                RowHasher,
                                                                decltype(window_extent)>;
  int grid_size =
    find_grid_size(compute_mapping_indices_fn_ptr, block_size, num_input_rows, num_sms);

  // 'local_mapping_index' maps from the global row index of the input table to the row index of
  // the local pre-aggregate table
  rmm::device_uvector<cudf::size_type> local_mapping_index(num_input_rows, stream);

  // 'global_mapping_index' maps from  the local pre-aggregate table to the row index of
  // global aggregate table
  rmm::device_uvector<cudf::size_type> global_mapping_index(grid_size * shared_set_num_elements,
                                                            stream);
  rmm::device_uvector<cudf::size_type> block_cardinality(grid_size, stream);

  rmm::device_scalar<bool> direct_aggregations(false, stream);

  compute_mapping_indices<shared_set_ref_type, shared_set_num_elements, cardinality_threshold>
    <<<grid_size, block_size, 0, stream>>>(global_set_ref,
                                           num_input_rows,
                                           window_extent,
                                           empty_key_sentinel,
                                           d_key_equal,
                                           d_row_hash,
                                           local_mapping_index.data(),
                                           global_mapping_index.data(),
                                           block_cardinality.data(),
                                           direct_aggregations.data(),
                                           static_cast<bitmask_type*>(row_bitmask.data()),
                                           skip_key_rows_with_nulls);

  stream.synchronize();

  // 'populated_keys' contains inserted row_indices (keys) of global hash set
  rmm::device_uvector<cudf::size_type> populated_keys(keys.num_rows(), stream);

  // TODO: change
  cudf::table sparse_table = create_sparse_results_table(flattened_values, agg_kinds, stream);

  auto d_sparse_table = cudf::mutable_table_device_view::create(sparse_table, stream);

  // TODO: rearrange arguments
  // TODO: too many arguments, consider making functors?
  launch_compute_aggregates<shared_set_num_elements, cardinality_threshold>(
    block_size,
    grid_size,
    num_sms,
    local_mapping_index.data(),
    global_mapping_index.data(),
    block_cardinality.data(),
    *d_values,
    *d_sparse_table,
    num_input_rows,
    d_aggs.data(),
    stream,
    static_cast<bitmask_type*>(row_bitmask.data()),
    skip_key_rows_with_nulls);

  if (direct_aggregations.value(stream)) {
    int stride = block_size * grid_size;
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator(0),
                       keys.num_rows(),
                       compute_direct_aggregates{global_set_ref,
                                                 *d_values,
                                                 *d_sparse_table,
                                                 d_aggs.data(),
                                                 static_cast<bitmask_type*>(row_bitmask.data()),
                                                 skip_key_rows_with_nulls,
                                                 block_cardinality.data(),
                                                 stride,
                                                 block_size,
                                                 cardinality_threshold});
    extract_populated_keys(global_set, populated_keys, stream);
  }

  auto sparse_result_cols = sparse_table.release();
  for (size_t i = 0; i < aggs.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    sparse_results->add_result(
      flattened_values.column(i), *aggs[i], std::move(sparse_result_cols[i]));
  }

  // TODO: fix later
  // return populated_keys;
}

/**
 * @brief Computes groupby using hash table.
 *
 * First, we create a hash table that stores the indices of unique rows in
 * `keys`. The upper limit on the number of values in this map is the number
 * of rows in `keys`.
 *
 * To store the results of aggregations, we create temporary sparse columns
 * which have the same size as input value columns. Using the hash map, we
 * determine the location within the sparse column to write the result of the
 * aggregation into.
 *
 * The sparse column results of all aggregations are stored into the cache
 * `sparse_results`. This enables the use of previously calculated results in
 * other aggregations.
 *
 * All the aggregations which can be computed in a single pass are computed
 * first, in a combined kernel. Then using these results, aggregations that
 * require multiple passes, will be computed.
 *
 * Finally, using the hash map, we generate a vector of indices of populated
 * values in sparse result columns. Then, for each aggregation originally
 * requested in `requests`, we gather sparse results into a column of dense
 * results using the aforementioned index vector. Dense results are stored into
 * the in/out parameter `cache`.
 */
std::unique_ptr<table> groupby(table_view const& keys,
                               host_span<aggregation_request const> requests,
                               cudf::detail::result_cache* cache,
                               bool const keys_have_nulls,
                               null_policy const include_null_keys,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  auto const num_keys            = keys.num_rows();
  auto const null_keys_are_equal = null_equality::EQUAL;
  auto const has_null            = nullate::DYNAMIC{cudf::has_nested_nulls(keys)};

  auto preprocessed_keys = cudf::experimental::row::hash::preprocessed_table::create(keys, stream);
  auto const comparator  = cudf::experimental::row::equality::self_comparator{preprocessed_keys};
  auto const row_hash    = cudf::experimental::row::hash::row_hasher{std::move(preprocessed_keys)};
  auto const d_row_hash  = row_hash.device_hasher(has_null);

  // Cache of sparse results where the location of aggregate value in each
  // column is indexed by the hash set
  cudf::detail::result_cache sparse_results(requests.size());

  auto const comparator_helper = [&](auto const d_key_equal) {
    auto empty_key_sentinel = cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL};
    auto const set          = cuco::static_set{num_keys,
                                      0.5,  // desired load factor
                                      empty_key_sentinel,
                                      d_key_equal,
                                      probing_scheme_type{d_row_hash},
                                      cuco::thread_scope_device,
                                      cuco::storage<1>{},
                                      cudf::detail::cuco_allocator{stream},
                                      stream.value()};

    // Compute all single pass aggs first
    compute_single_pass_set_aggs(keys,
                                 requests,
                                 &sparse_results,
                                 set,
                                 keys_have_nulls,
                                 include_null_keys,
                                 stream,
                                 empty_key_sentinel,
                                 d_key_equal,
                                 d_row_hash);

    // Extract the populated indices from the hash set and create a gather map.
    // Gathering using this map from sparse results will give dense results.
    auto gather_map = extract_populated_keys(set, keys.num_rows(), stream);

    // Compact all results from sparse_results and insert into cache
    sparse_to_dense_results(keys,
                            requests,
                            &sparse_results,
                            cache,
                            gather_map,
                            set.ref(cuco::find),
                            keys_have_nulls,
                            include_null_keys,
                            stream,
                            mr);

    return cudf::detail::gather(keys,
                                gather_map,
                                out_of_bounds_policy::DONT_CHECK,
                                cudf::detail::negative_index_policy::NOT_ALLOWED,
                                stream,
                                mr);
  };

  if (cudf::detail::has_nested_columns(keys)) {
    // TODO: fix
    auto const d_key_equal = comparator.equal_to<true>(has_null, null_keys_are_equal);
    return comparator_helper(d_key_equal);
  } else {
    auto const d_key_equal = comparator.equal_to<false>(has_null, null_keys_are_equal);
    return comparator_helper(d_key_equal);
  }
}

}  // namespace

/**
 * @brief Indicates if a set of aggregation requests can be satisfied with a
 * hash-based groupby implementation.
 *
 * @param requests The set of columns to aggregate and the aggregations to
 * perform
 * @return true A hash-based groupby should be used
 * @return false A hash-based groupby should not be used
 */
bool can_use_hash_groupby(host_span<aggregation_request const> requests)
{
  return std::all_of(requests.begin(), requests.end(), [](aggregation_request const& r) {
    auto const v_type = is_dictionary(r.values.type())
                          ? cudf::dictionary_column_view(r.values).keys().type()
                          : r.values.type();

    // Currently, input values (not keys) of STRUCT and LIST types are not supported in any of
    // hash-based aggregations. For those situations, we fallback to sort-based aggregations.
    if (v_type.id() == type_id::STRUCT or v_type.id() == type_id::LIST) { return false; }

    return std::all_of(r.aggregations.begin(), r.aggregations.end(), [v_type](auto const& a) {
      return cudf::has_atomic_support(cudf::detail::target_type(v_type, a->kind)) and
             is_hash_aggregation(a->kind);
    });
  });
}

// Hash-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  null_policy include_null_keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::detail::result_cache cache(requests.size());

  std::unique_ptr<table> unique_keys =
    groupby(keys, requests, &cache, cudf::has_nulls(keys), include_null_keys, stream, mr);

  return std::pair(std::move(unique_keys), extract_results(requests, cache, stream, mr));
}
}  // namespace hash
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
