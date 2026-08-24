/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "lfs/core/warp_reduce.cuh"
#include "lfs/kernels/l1_loss.cuh"
#include <type_traits>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {
    namespace {
        constexpr float UINT8_TO_FLOAT = 1.0f / 255.0f;

        template <typename T>
        __device__ __forceinline__ float target_value(const T* data, size_t idx) {
            if constexpr (std::is_same_v<std::remove_cv_t<T>, uint8_t>) {
                return static_cast<float>(data[idx]) * UINT8_TO_FLOAT;
            } else {
                return static_cast<float>(data[idx]);
            }
        }
    } // namespace

    // Fused kernel: computes gradient and accumulates sum(abs(diff)) in single pass
    // OPTIMIZED: Uses warp-level reductions (5-10× faster than CUB BlockReduce!)
    template <typename TargetT>
    __global__ void fused_l1_kernel(
        const float* __restrict__ img1,
        const TargetT* __restrict__ img2,
        float* __restrict__ grad_out,
        float* __restrict__ partial_sums,
        size_t N,
        float grad_scale,
        int band_h = 0,
        int band_w = 0,
        int band_top_rows = 0,
        float band_w_top = 1.0f,
        float band_w_rest = 1.0f) {

        // Thread-local sum
        float local_sum = 0.0f;

        // Grid-stride loop for coalesced memory access
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += blockDim.x * gridDim.x) {

            float diff = img1[idx] - target_value(img2, idx);
            float abs_diff = fabsf(diff);

            // Round 23 D1: row-band weight for the loss contribution and the
            // gradient alike. Inactive when band_top_rows == 0.
            float band_weight = 1.0f;
            if (band_top_rows > 0) {
                const int h = static_cast<int>((idx / static_cast<size_t>(band_w)) %
                                               static_cast<size_t>(band_h));
                band_weight = (h < band_top_rows) ? band_w_top : band_w_rest;
                abs_diff *= band_weight;
            }

            // Accumulate for loss
            local_sum += abs_diff;

            // Store gradient: sign(diff) * grad_scale
            // NOTE: sign(0) = 0 to match PyTorch behavior
            // copysignf returns ±grad_scale even for 0.0, which is wrong
            float grad = (diff > 0.0f) ? grad_scale : ((diff < 0.0f) ? -grad_scale : 0.0f);
            grad_out[idx] = band_top_rows > 0 ? grad * band_weight : grad;
        }

        // Block-level warp reduction (tiny-cuda-nn style - much faster!)
        local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);

        // First thread writes block result
        if (threadIdx.x == 0) {
            partial_sums[blockIdx.x] = local_sum;
        }
    }

    // Final reduction kernel (handles any number of blocks)
    // OPTIMIZED: Uses warp-level reductions (5-10× faster than CUB BlockReduce!)
    __global__ void final_reduce_kernel(
        const float* __restrict__ partial_sums,
        float* __restrict__ result,
        int num_blocks,
        float norm_factor) {

        // Grid-stride loop to handle more than blockDim.x partial sums
        float sum = 0.0f;
        for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
            sum += partial_sums[i];
        }

        // Block-level warp reduction (tiny-cuda-nn style - much faster!)
        sum = lfs::core::warp_ops::block_reduce_sum(sum);

        if (threadIdx.x == 0) {
            result[0] = sum * norm_factor;
        }
    }

    template <typename TargetT>
    void launch_fused_l1_loss_impl(
        const float* img1,
        const TargetT* img2,
        float* grad_out,
        float* loss_out,
        float* temp_buffer,
        size_t N,
        cudaStream_t stream,
        int band_image_h,
        int band_image_w,
        int band_top_rows,
        float band_w_top,
        float band_w_rest) {
        LFS_ASSERT_MSG(img1 != nullptr && img2 != nullptr && grad_out != nullptr &&
                           loss_out != nullptr && temp_buffer != nullptr,
                       "Fused L1 loss pointers must be non-null");
        LFS_ASSERT_MSG(N > 0, "Fused L1 loss requires at least one element");
        stream = resolve_stream(stream);

        const int block_size = 256;
        const int num_blocks = std::min((N + block_size - 1) / block_size, size_t(1024));

        float grad_scale = 1.0f / static_cast<float>(N);

        // Launch fused kernel
        fused_l1_kernel<TargetT><<<num_blocks, block_size, 0, stream>>>(
            img1, img2, grad_out, temp_buffer, N, grad_scale,
            band_image_h, band_image_w, band_top_rows, band_w_top, band_w_rest);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.l1.fused");

        // Launch final reduction (normalize by N for mean)
        float norm_factor = 1.0f / static_cast<float>(N);
        final_reduce_kernel<<<1, block_size, 0, stream>>>(
            temp_buffer, loss_out, num_blocks, norm_factor);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.l1.final_reduce");
    }

    void launch_fused_l1_loss(
        const float* img1,
        const float* img2,
        float* grad_out,
        float* loss_out,
        float* temp_buffer,
        size_t N,
        cudaStream_t stream,
        int band_image_h,
        int band_image_w,
        int band_top_rows,
        float band_w_top,
        float band_w_rest) {
        stream = resolve_stream(stream);
        launch_fused_l1_loss_impl(img1, img2, grad_out, loss_out, temp_buffer, N, stream,
                                  band_image_h, band_image_w, band_top_rows,
                                  band_w_top, band_w_rest);
    }

    void launch_fused_l1_loss(
        const float* img1,
        const uint8_t* img2,
        float* grad_out,
        float* loss_out,
        float* temp_buffer,
        size_t N,
        cudaStream_t stream,
        int band_image_h,
        int band_image_w,
        int band_top_rows,
        float band_w_top,
        float band_w_rest) {
        stream = resolve_stream(stream);
        launch_fused_l1_loss_impl(img1, img2, grad_out, loss_out, temp_buffer, N, stream,
                                  band_image_h, band_image_w, band_top_rows,
                                  band_w_top, band_w_rest);
    }

} // namespace lfs::training::kernels
