#include <vector>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <cuda_fp16.h>

#include "../tester/utils.h"

namespace {

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float toFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half fromFloat<half>(float value) {
  return __float2half_rn(value);
}

__device__ __forceinline__ float warpReduceSum(float value) {
  constexpr unsigned int kFullMask = 0xffffffffu;
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(kFullMask, value, offset);
  }
  return value;
}

template <typename T, int VecWidth>
struct RmsNormIO;

template <typename T>
struct RmsNormIO<T, 1> {
  __device__ __forceinline__ static float squareSum(const T* input,
                                                     size_t index) {
    const float value = toFloat(input[index]);
    return value * value;
  }

  __device__ __forceinline__ static void normalizeStore(
      const T* input, const T* weight, T* output, size_t input_index,
      size_t weight_index, float inv_rms) {
    const float value =
        toFloat(input[input_index]) * inv_rms * toFloat(weight[weight_index]);
    output[input_index] = fromFloat<T>(value);
  }
};

template <>
struct RmsNormIO<float, 4> {
  __device__ __forceinline__ static float squareSum(const float* input,
                                                     size_t index) {
    const float4 value = reinterpret_cast<const float4*>(input)[index];
    float sum = value.x * value.x;
    sum += value.y * value.y;
    sum += value.z * value.z;
    sum += value.w * value.w;
    return sum;
  }

  __device__ __forceinline__ static void normalizeStore(
      const float* input, const float* weight, float* output,
      size_t input_index, size_t weight_index, float inv_rms) {
    const float4 value = reinterpret_cast<const float4*>(input)[input_index];
    const float4 scale = reinterpret_cast<const float4*>(weight)[weight_index];
    reinterpret_cast<float4*>(output)[input_index] = make_float4(
        value.x * inv_rms * scale.x, value.y * inv_rms * scale.y,
        value.z * inv_rms * scale.z, value.w * inv_rms * scale.w);
  }
};

template <>
struct RmsNormIO<half, 2> {
  __device__ __forceinline__ static float squareSum(const half* input,
                                                     size_t index) {
    const half2 value = reinterpret_cast<const half2*>(input)[index];
    const float2 pair = __half22float2(value);
    return pair.x * pair.x + pair.y * pair.y;
  }

  __device__ __forceinline__ static void normalizeStore(
      const half* input, const half* weight, half* output, size_t input_index,
      size_t weight_index, float inv_rms) {
    const float2 value =
        __half22float2(reinterpret_cast<const half2*>(input)[input_index]);
    const float2 scale =
        __half22float2(reinterpret_cast<const half2*>(weight)[weight_index]);
    reinterpret_cast<half2*>(output)[input_index] = __floats2half2_rn(
        value.x * inv_rms * scale.x, value.y * inv_rms * scale.y);
  }
};

template <typename T, int VecWidth>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t rows, size_t hidden_dim, float eps) {
  using IO = RmsNormIO<T, VecWidth>;
  const size_t items_per_row = hidden_dim / VecWidth;
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp_id = threadIdx.x / warpSize;
  const int warp_count = blockDim.x / warpSize;
  __shared__ float warp_sums[32];
  __shared__ float inv_rms;

  // A block handles one row at a time. A grid-stride loop keeps the launch
  // valid even when rows is larger than the portable grid-size limit used by
  // the host wrapper.
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const size_t row_offset = row * items_per_row;
    float square_sum = 0.0f;

    // Accumulating half inputs in float avoids the severe precision loss of a
    // half reduction and matches the usual mixed-precision RMSNorm behavior.
    for (size_t item = threadIdx.x; item < items_per_row;
         item += blockDim.x) {
      square_sum += IO::squareSum(input, row_offset + item);
    }

    square_sum = warpReduceSum(square_sum);

    if (lane == 0) {
      warp_sums[warp_id] = square_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
      float block_sum = lane < warp_count ? warp_sums[lane] : 0.0f;
      block_sum = warpReduceSum(block_sum);
      if (lane == 0) {
        inv_rms = rsqrtf(block_sum / static_cast<float>(hidden_dim) + eps);
      }
    }
    __syncthreads();

    for (size_t item = threadIdx.x; item < items_per_row;
         item += blockDim.x) {
      IO::normalizeStore(input, weight, output, row_offset + item, item,
                         inv_rms);
    }
    __syncthreads();
  }
}

constexpr int kBlockCandidates[] = {32, 64, 128, 256, 512, 1024};

template <typename T, int VecWidth>
struct RmsNormOccupancyInfo {
  int sm_count = 0;
  int max_threads_per_block = 0;
  int active_blocks_per_sm[6] = {};

  RmsNormOccupancyInfo() {
    int device = 0;
    cudaDeviceProp properties{};
    RUNTIME_CHECK(cudaGetDevice(&device));
    RUNTIME_CHECK(cudaGetDeviceProperties(&properties, device));
    sm_count = properties.multiProcessorCount;
    max_threads_per_block = properties.maxThreadsPerBlock;

    for (int i = 0; i < 6; ++i) {
      if (kBlockCandidates[i] <= max_threads_per_block) {
        RUNTIME_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm[i], rmsNormKernel<T, VecWidth>,
            kBlockCandidates[i], 0));
      }
    }
  }
};

template <typename T, int VecWidth>
int rmsNormBlockSize(size_t rows, size_t hidden_dim) {
  // The occupancy information is initialized once per data/vector type during
  // the warm-up call. This adapts to the evaluator GPU without adding runtime
  // API overhead to every profiled invocation.
  static const RmsNormOccupancyInfo<T, VecWidth> occupancy;
  const size_t items_per_row = hidden_dim / VecWidth;

  int best_threads = 32;
  uint64_t best_cost = std::numeric_limits<uint64_t>::max();
  for (int i = 0; i < 6; ++i) {
    const int threads = kBlockCandidates[i];
    const int active_blocks = occupancy.active_blocks_per_sm[i];
    if (threads > occupancy.max_threads_per_block || active_blocks == 0) {
      continue;
    }

    // More threads than the next power-of-two covering the row only add idle
    // warps and reduction work.
    if (threads > 32 && static_cast<size_t>(threads / 2) >= items_per_row) {
      continue;
    }

    const uint64_t resident_blocks =
        static_cast<uint64_t>(occupancy.sm_count) * active_blocks;
    const uint64_t waves =
        (static_cast<uint64_t>(rows) + resident_blocks - 1) / resident_blocks;
    const uint64_t iterations =
        (static_cast<uint64_t>(items_per_row) + threads - 1) / threads;
    const uint64_t cost = waves * iterations;

    if (cost < best_cost || (cost == best_cost && threads < best_threads)) {
      best_cost = cost;
      best_threads = threads;
    }
  }
  return best_threads;
}

template <typename T>
class DeviceBufferCache {
 public:
  ~DeviceBufferCache() {
    // Destructors must not throw or terminate the process during CUDA runtime
    // shutdown. All operational CUDA calls are still checked in ensure().
    if (io_buffer_ != nullptr) {
      cudaFree(io_buffer_);
    }
    if (weight_ != nullptr) {
      cudaFree(weight_);
    }
  }

  void ensure(size_t element_count, size_t hidden_dim) {
    // Keep the output half of the combined allocation 16-byte aligned even
    // when a larger, previously cached request had an odd element count.
    constexpr size_t kAlignmentElements = 16 / sizeof(T);
    const size_t aligned_element_count =
        ((element_count + kAlignmentElements - 1) / kAlignmentElements) *
        kAlignmentElements;

    if (aligned_element_count > io_capacity_) {
      if (io_buffer_ != nullptr) {
        RUNTIME_CHECK(cudaFree(io_buffer_));
        io_buffer_ = nullptr;
      }
      RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&io_buffer_),
                               2 * aligned_element_count * sizeof(T)));
      io_capacity_ = aligned_element_count;
    }

    if (hidden_dim > weight_capacity_) {
      if (weight_ != nullptr) {
        RUNTIME_CHECK(cudaFree(weight_));
        weight_ = nullptr;
      }
      RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&weight_),
                               hidden_dim * sizeof(T)));
      weight_capacity_ = hidden_dim;
    }
  }

  T* input() const { return io_buffer_; }
  T* output() const { return io_buffer_ + io_capacity_; }
  T* weight() const { return weight_; }

 private:
  T* io_buffer_ = nullptr;
  T* weight_ = nullptr;
  size_t io_capacity_ = 0;
  size_t weight_capacity_ = 0;
};

template <typename T>
struct PreferredVectorWidth;

template <>
struct PreferredVectorWidth<float> {
  static constexpr int value = 4;
};

template <>
struct PreferredVectorWidth<half> {
  static constexpr int value = 2;
};

template <typename T, int VecWidth>
void launchRmsNorm(const T* input, const T* weight, T* output, size_t rows,
                   size_t hidden_dim, float eps) {
  const int threads = rmsNormBlockSize<T, VecWidth>(rows, hidden_dim);
  constexpr size_t kMaxPortableGridX = 65535;
  const unsigned int blocks =
      static_cast<unsigned int>(std::min(rows, kMaxPortableGridX));
  rmsNormKernel<T, VecWidth><<<blocks, threads>>>(
      input, weight, output, rows, hidden_dim, eps);
}

template <typename T>
__global__ void flashAttentionKernel(
    const T* query, const T* key, const T* value, T* output, int batch_size,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads,
    int head_dim, bool is_causal) {
  extern __shared__ float attention_shared[];
  float* const query_shared = attention_shared;
  float* const scores = query_shared + head_dim;

  __shared__ float softmax_sum;
  __shared__ float softmax_max;

  const size_t task_count = static_cast<size_t>(batch_size) *
                            target_seq_len * query_heads;
  // Match the reference instruction order: sqrt -> reciprocal once, followed
  // by a multiply for every scaled Q*K score.
  const float score_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int queries_per_kv_head = query_heads / kv_heads;

  // Each block computes one complete output vector. Scores are kept only for
  // the current query in shared memory, so no global [target, source] matrix is
  // materialized.
  for (size_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    size_t task_index = task;
    const int query_head = task_index % query_heads;
    task_index /= query_heads;
    const int target_index = task_index % target_seq_len;
    const int batch_index = task_index / target_seq_len;
    const int kv_head = query_head / queries_per_kv_head;

    const size_t query_offset =
        ((static_cast<size_t>(batch_index) * target_seq_len + target_index) *
             query_heads +
         query_head) *
        head_dim;

    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      query_shared[dim] = toFloat(query[query_offset + dim]);
    }
    __syncthreads();

    // PyTorch's causal SDPA mask is the upper-left aligned lower triangle.
    // Consequently, for non-square attention query i may see keys [0, i].
    int source_end = src_seq_len;
    if (is_causal && target_index + 1 < source_end) {
      source_end = target_index + 1;
    }

    // Parallelize across source positions, but accumulate every Q*K dot
    // product in increasing dimension order. This mirrors the reference
    // kernel's FMA order and avoids strict-float-tolerance differences caused
    // by a tree reduction over head_dim.
    for (int source_index = threadIdx.x; source_index < source_end;
         source_index += blockDim.x) {
      const size_t kv_offset =
          ((static_cast<size_t>(batch_index) * src_seq_len + source_index) *
               kv_heads +
           kv_head) *
          head_dim;

      float dot = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        dot = fmaf(query_shared[dim], toFloat(key[kv_offset + dim]), dot);
      }
      scores[source_index] = dot * score_scale;
    }
    __syncthreads();

    // The reference performs max and denominator accumulation in source order.
    // One thread reproduces that order while all other threads are free of
    // block-wide synchronization inside the source loop above.
    constexpr bool kIsFloat = std::is_same<T, float>::value;
    const bool recompute_output_scores =
        kIsFloat && is_causal && head_dim == 32;
    if (threadIdx.x == 0) {
      float max_score = -FLT_MAX;
      for (int source_index = 0; source_index < source_end; ++source_index) {
        max_score = fmaxf(max_score, scores[source_index]);
      }

      float sum = 0.0f;
      for (int source_index = 0; source_index < source_end; ++source_index) {
        const float probability = expf(scores[source_index] - max_score);
        if (!recompute_output_scores) {
          scores[source_index] = probability;
        }
        sum += probability;
      }

      softmax_sum = sum;
      softmax_max = max_score;
    }
    __syncthreads();

    if (recompute_output_scores) {
      // For strict float D=32 causal cases, recompute the output-pass score so
      // dot*scale flows directly into expf, as in the reference kernel. Source
      // positions are independent and remain parallel across the block.
      for (int source_index = threadIdx.x; source_index < source_end;
           source_index += blockDim.x) {
        const size_t kv_offset =
            ((static_cast<size_t>(batch_index) * src_seq_len + source_index) *
                 kv_heads +
             kv_head) *
            head_dim;
        float dot = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
          dot = fmaf(query_shared[dim], toFloat(key[kv_offset + dim]), dot);
        }
        scores[source_index] = expf(dot * score_scale - softmax_max);
      }
      __syncthreads();
    }

    const float inverse_sum = 1.0f / softmax_sum;
    const size_t value_stride = static_cast<size_t>(kv_heads) * head_dim;
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      size_t value_index =
          ((static_cast<size_t>(batch_index) * src_seq_len) * kv_heads +
           kv_head) *
              head_dim +
          dim;
      float accumulator = 0.0f;
      for (int source_index = 0; source_index < source_end; ++source_index) {
        // Normalize each probability before the V accumulation. This is not
        // algebraically interchangeable with multiplying the final
        // accumulator by inverse_sum in float arithmetic, and mirrors the
        // reference kernel's mul + fma sequence.
        const float probability = scores[source_index] * inverse_sum;
        accumulator =
            fmaf(probability, toFloat(value[value_index]), accumulator);
        value_index += value_stride;
      }
      output[query_offset + dim] = fromFloat<T>(accumulator);
    }
    // Required before this grid-stride block reuses shared memory for a new
    // output vector.
    __syncthreads();
  }
}

int attentionBlockSize(int head_dim) {
  int threads = 32;
  while (threads < 256 && threads < head_dim) {
    threads *= 2;
  }
  return threads;
}

template <typename T>
class AttentionBufferCache {
 public:
  ~AttentionBufferCache() {
    if (query_ != nullptr) {
      cudaFree(query_);
    }
    if (key_ != nullptr) {
      cudaFree(key_);
    }
    if (value_ != nullptr) {
      cudaFree(value_);
    }
    if (output_ != nullptr) {
      cudaFree(output_);
    }
  }

  void ensure(size_t query_count, size_t key_value_count) {
    ensureAllocation(query_, query_capacity_, query_count);
    ensureAllocation(output_, output_capacity_, query_count);
    ensureAllocation(key_, key_capacity_, key_value_count);
    ensureAllocation(value_, value_capacity_, key_value_count);
  }

  T* query() const { return query_; }
  T* key() const { return key_; }
  T* value() const { return value_; }
  T* output() const { return output_; }

 private:
  static void ensureAllocation(T*& pointer, size_t& capacity,
                               size_t required_count) {
    if (required_count <= capacity) {
      return;
    }
    if (pointer != nullptr) {
      RUNTIME_CHECK(cudaFree(pointer));
      pointer = nullptr;
    }
    RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer),
                             required_count * sizeof(T)));
    capacity = required_count;
  }

  T* query_ = nullptr;
  T* key_ = nullptr;
  T* value_ = nullptr;
  T* output_ = nullptr;
  size_t query_capacity_ = 0;
  size_t key_capacity_ = 0;
  size_t value_capacity_ = 0;
  size_t output_capacity_ = 0;
};

}  // namespace

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  if (rows == 0 || hidden_dim == 0) {
    return;
  }

  const size_t element_count = rows * hidden_dim;
  const size_t input_bytes = element_count * sizeof(T);
  const size_t weight_bytes = hidden_dim * sizeof(T);
  static thread_local DeviceBufferCache<T> buffers;
  buffers.ensure(element_count, hidden_dim);
  T* const d_input = buffers.input();
  T* const d_weight = buffers.weight();
  T* const d_output = buffers.output();

  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes,
                           cudaMemcpyHostToDevice));

  constexpr int kVectorWidth = PreferredVectorWidth<T>::value;
  if (hidden_dim % kVectorWidth == 0) {
    launchRmsNorm<T, kVectorWidth>(d_input, d_weight, d_output, rows,
                                   hidden_dim, eps);
  } else {
    launchRmsNorm<T, 1>(d_input, d_weight, d_output, rows, hidden_dim, eps);
  }
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, input_bytes,
                           cudaMemcpyDeviceToHost));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }

  const size_t query_count = static_cast<size_t>(batch_size) *
                             target_seq_len * query_heads * head_dim;
  const size_t key_value_count = static_cast<size_t>(batch_size) *
                                 src_seq_len * kv_heads * head_dim;
  const size_t query_bytes = query_count * sizeof(T);
  const size_t key_value_bytes = key_value_count * sizeof(T);

  static thread_local AttentionBufferCache<T> buffers;
  buffers.ensure(query_count, key_value_count);
  RUNTIME_CHECK(cudaMemcpy(buffers.query(), h_q.data(), query_bytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(buffers.key(), h_k.data(), key_value_bytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(buffers.value(), h_v.data(), key_value_bytes,
                           cudaMemcpyHostToDevice));

  const size_t task_count = static_cast<size_t>(batch_size) *
                            target_seq_len * query_heads;
  constexpr size_t kMaxPortableGridX = 65535;
  const unsigned int blocks =
      static_cast<unsigned int>(std::min(task_count, kMaxPortableGridX));
  const int threads = attentionBlockSize(head_dim);
  const int score_capacity =
      is_causal ? std::min(src_seq_len, target_seq_len) : src_seq_len;
  const size_t shared_bytes =
      (static_cast<size_t>(head_dim) + score_capacity) * sizeof(float);
  flashAttentionKernel<T><<<blocks, threads, shared_bytes>>>(
      buffers.query(), buffers.key(), buffers.value(), buffers.output(),
      batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim,
      is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), buffers.output(), query_bytes,
                           cudaMemcpyDeviceToHost));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
