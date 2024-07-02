/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#pragma once

#include "groupby/hash/row_aggregator.cuh"
#include "multi_pass_kernels.cuh"

#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/groupby.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/utilities/bit.hpp>

#include <cuco/static_set.cuh>
#include <thrust/pair.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace hash {

// TODO: similar to `contains_table`, using larger CG size like 2 or 4 for nested
// types and `cg_size = 1`for flat data to improve performance
using probing_scheme_type = cuco::linear_probing<
  1,  ///< Number of threads used to handle each input key
  cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                   cudf::nullate::DYNAMIC>>;

/**
 * @brief Computes single-pass aggregations and store results into a sparse `output_values` table,
 * and populate `set` with indices of unique keys
 *
 * The hash set is built by inserting every row index `i` from the `keys` and `values` tables. If
 * the index was not present in the set, insert they index and then copy it to the output. If the
 * key was already present in the set, then the inserted index is aggregated with the existing row.
 * This aggregation is done for every element `j` in the row by applying aggregation operation `j`
 * between the new and existing element.
 *
 * Instead of storing the entire rows from `input_keys` and `input_values` in
 * the hashset, we instead store the row indices. For example, when inserting
 * row at index `i` from `input_keys` into the hash set, the value `i` is what
 * gets stored for the hash set's "key". It is assumed the `set` was constructed
 * with a custom comparator that uses these row indices to check for equality
 * between key rows. For example, comparing two keys `k0` and `k1` will compare
 * the two rows `input_keys[k0] ?= input_keys[k1]`
 *
 * The exact size of the result is not known a priori, but can be upper bounded
 * by the number of rows in `input_keys` & `input_values`. Therefore, it is
 * assumed `output_values` has sufficient storage for an equivalent number of
 * rows. In this way, after all rows are aggregated, `output_values` will likely
 * be "sparse", meaning that not all rows contain the result of an aggregation.
 *
 * @tparam SetType The type of the hash set device ref
 */
template <typename SetType>
struct compute_single_pass_aggs_fn {
  SetType set;
  table_device_view input_values;
  mutable_table_device_view output_values;
  aggregation::Kind const* __restrict__ aggs;
  bitmask_type const* __restrict__ row_bitmask;
  bool skip_rows_with_nulls;

  /**
   * @brief Construct a new compute_single_pass_aggs_fn functor object
   *
   * @param set_ref Hash set object to insert key,value pairs into.
   * @param input_values The table whose rows will be aggregated in the values
   * of the hash set
   * @param output_values Table that stores the results of aggregating rows of
   * `input_values`.
   * @param aggs The set of aggregation operations to perform across the
   * columns of the `input_values` rows
   * @param row_bitmask Bitmask where bit `i` indicates the presence of a null
   * value in row `i` of input keys. Only used if `skip_rows_with_nulls` is `true`
   * @param skip_rows_with_nulls Indicates if rows in `input_keys` containing
   * null values should be skipped. It `true`, it is assumed `row_bitmask` is a
   * bitmask where bit `i` indicates the presence of a null value in row `i`.
   */
  compute_single_pass_aggs_fn(SetType set,
                              table_device_view input_values,
                              mutable_table_device_view output_values,
                              aggregation::Kind const* aggs,
                              bitmask_type const* row_bitmask,
                              bool skip_rows_with_nulls)
    : set(set),
      input_values(input_values),
      output_values(output_values),
      aggs(aggs),
      row_bitmask(row_bitmask),
      skip_rows_with_nulls(skip_rows_with_nulls)
  {
  }

  __device__ void operator()(size_type i)
  {
    if (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, i)) {
      auto const result = set.insert_and_find(i);

      cudf::detail::aggregate_row<true, true>(output_values, *result.first, input_values, i, aggs);
    }
  }
};

// TODO: they might have something similar
template <typename type, std::enable_if_t<std::is_integral_v<type>>* = nullptr>
constexpr __device__ __host__ type divide_round_up(type dividend, type divisor)
{
  assert(divisor != 0);

  return dividend / divisor + (dividend % divisor != 0);
}

__device__ __host__ size_t round_to_multiple_of_8(size_t num)
{
  return (num / 8 + (num % 8 != 0)) * 8;
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

  bool process_row = cur_idx < num_input_rows &&
                     (not skip_rows_with_nulls || cudf::bit_is_set(row_bitmask, cur_idx));

  if (process_row) {
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

  // Syncing the thread block is needed so that updates in `local_mapping_index` are visible to all
  // threads in the thread block.
  __syncthreads();

  if (process_row && !inserted) { local_mapping_index[cur_idx] = local_mapping_index[result_idx]; }
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

  int num_loops =
    util::div_rounding_up_safe(num_input_rows, (cudf::size_type)(blockDim.x * gridDim.x));
  auto end_idx = num_loops * blockDim.x * gridDim.x;

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
    int valid_col_size  = cudf::detail::round_up_pow2(sizeof(bool) * cardinality, 8UL);
    while ((bytes_allocated < total_agg_size) && (col_end < num_input_cols)) {
      int next_col_size = cudf::detail::round_up_pow2(
        sizeof(output_values.column(col_end).type()) * cardinality, 8UL);
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
                                                  initialize_shmem{},
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
                                                    shmem_element_aggregator{},
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
                                                  gmem_element_aggregator{},
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

}  // namespace hash
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
