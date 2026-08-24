/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "errmap_kernels.hpp"
#include "kernel_stream.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace lfs::training::kernels {

    namespace {
        // 7-tap Gaussian, sigma = 1.5: w[i] = exp(-i^2 / (2 * 1.5^2)), normalized
        // so the tap sum is exactly 1.0f in float arithmetic.
        __constant__ float ERRMAP_GAUSS7[7] = {
            0.036632845f, 0.111280758f, 0.216745321f, 0.270682149f,
            0.216745321f, 0.111280758f, 0.036632845f};
    } // namespace

    __global__ void errmap_l1_residual_kernel(
        const float* __restrict__ render,
        const float* __restrict__ gt,
        const int channels,
        const int height,
        const int width,
        float* __restrict__ out) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        const int plane = height * width;
        const int idx = y * width + x;
        float acc = 0.0f;
        for (int c = 0; c < channels; ++c) {
            acc += fabsf(render[c * plane + idx] - gt[c * plane + idx]);
        }
        out[idx] = acc / static_cast<float>(channels);
    }

    __global__ void errmap_luma_rec601_kernel(
        const float* __restrict__ img,
        const int channels,
        const int height,
        const int width,
        float* __restrict__ luma) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        const int plane = height * width;
        const int idx = y * width + x;
        if (channels >= 3) {
            luma[idx] = 0.299f * img[idx] +
                        0.587f * img[idx + plane] +
                        0.114f * img[idx + 2 * plane];
        } else {
            luma[idx] = img[idx];
        }
    }

    __global__ void errmap_gauss7_pass_kernel(
        const float* __restrict__ in,
        const int height,
        const int width,
        const int horizontal,
        float* __restrict__ out) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        float acc = 0.0f;
#pragma unroll
        for (int t = -3; t <= 3; ++t) {
            if (horizontal) {
                const int xi = min(max(x + t, 0), width - 1);
                acc += ERRMAP_GAUSS7[t + 3] * in[y * width + xi];
            } else {
                const int yi = min(max(y + t, 0), height - 1);
                acc += ERRMAP_GAUSS7[t + 3] * in[yi * width + x];
            }
        }
        out[y * width + x] = acc;
    }

    __global__ void errmap_detail_deficit_kernel(
        const float* __restrict__ gt,
        const float* __restrict__ gt_blurred,
        const float* __restrict__ render,
        const float* __restrict__ render_blurred,
        const std::size_t n,
        float* __restrict__ out) {
        const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
        for (std::size_t i = idx; i < n; i += stride) {
            const float hf_gt = fabsf(gt[i] - gt_blurred[i]);
            const float hf_render = fabsf(render[i] - render_blurred[i]);
            out[i] = fmaxf(hf_gt - hf_render, 0.0f);
        }
    }

    __global__ void errmap_mix_kernel(
        const float* __restrict__ a,
        const float* __restrict__ b,
        const std::size_t n,
        float* __restrict__ out) {
        const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
        for (std::size_t i = idx; i < n; i += stride) {
            out[i] = 0.5f * a[i] + 0.5f * b[i];
        }
    }

    // ============================================================================
    // Launch functions
    // ============================================================================

    void launch_errmap_l1_residual(
        const float* d_render_chw,
        const float* d_gt_chw,
        const int channels,
        const int height,
        const int width,
        float* d_out_hw,
        cudaStream_t stream) {
        stream = resolve_stream(stream);
        const dim3 block(32, 8, 1);
        const dim3 grid((width + block.x - 1) / block.x,
                        (height + block.y - 1) / block.y);
        errmap_l1_residual_kernel<<<grid, block, 0, stream>>>(
            d_render_chw, d_gt_chw, channels, height, width, d_out_hw);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.errmap.l1_residual");
    }

    void launch_errmap_luma_rec601(
        const float* d_img_chw,
        const int channels,
        const int height,
        const int width,
        float* d_luma_hw,
        cudaStream_t stream) {
        stream = resolve_stream(stream);
        const dim3 block(32, 8, 1);
        const dim3 grid((width + block.x - 1) / block.x,
                        (height + block.y - 1) / block.y);
        errmap_luma_rec601_kernel<<<grid, block, 0, stream>>>(
            d_img_chw, channels, height, width, d_luma_hw);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.errmap.luma_rec601");
    }

    void launch_errmap_gauss7_pass(
        const float* d_in_hw,
        const int height,
        const int width,
        const bool horizontal,
        float* d_out_hw,
        cudaStream_t stream) {
        stream = resolve_stream(stream);
        const dim3 block(32, 8, 1);
        const dim3 grid((width + block.x - 1) / block.x,
                        (height + block.y - 1) / block.y);
        errmap_gauss7_pass_kernel<<<grid, block, 0, stream>>>(
            d_in_hw, height, width, horizontal ? 1 : 0, d_out_hw);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.errmap.gauss7_pass");
    }

    void launch_errmap_detail_deficit(
        const float* d_gt_hw,
        const float* d_gt_blurred_hw,
        const float* d_render_hw,
        const float* d_render_blurred_hw,
        const std::size_t n,
        float* d_out_hw,
        cudaStream_t stream) {
        stream = resolve_stream(stream);
        constexpr int block_size = 256;
        const int grid_size = static_cast<int>(
            std::min<std::size_t>((n + block_size - 1) / block_size, 4096));
        errmap_detail_deficit_kernel<<<grid_size, block_size, 0, stream>>>(
            d_gt_hw, d_gt_blurred_hw, d_render_hw, d_render_blurred_hw, n, d_out_hw);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.errmap.detail_deficit");
    }

    void launch_errmap_mix(
        const float* d_a_hw,
        const float* d_b_hw,
        const std::size_t n,
        float* d_out_hw,
        cudaStream_t stream) {
        stream = resolve_stream(stream);
        constexpr int block_size = 256;
        const int grid_size = static_cast<int>(
            std::min<std::size_t>((n + block_size - 1) / block_size, 4096));
        errmap_mix_kernel<<<grid_size, block_size, 0, stream>>>(d_a_hw, d_b_hw, n, d_out_hw);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.errmap.mix");
    }

} // namespace lfs::training::kernels
