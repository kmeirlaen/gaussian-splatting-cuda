/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Fused mean reduction for SSIM map with optional valid padding
     *
     * Computes mean(ssim_map) with optional cropping (5 pixels from each side)
     * in a single optimized pass without creating an intermediate cropped tensor.
     *
     * @param ssim_map Input SSIM map [N, C, H, W]
     * @param temp_buffer Temporary buffer for partial sums (min(1024, total_pixels/256) elements)
     * @param result_buffer Device buffer for result (1 element)
     * @param N Batch size
     * @param C Number of channels
     * @param H Height
     * @param W Width
     * @param apply_valid_padding If true, crop 5 pixels from each side
     * @param stream CUDA stream
     *
     * Round 23 diagnostic D1 (LFS_EXP_BAND_WEIGHT): when band_top_rows > 0 the
     * per-pixel SSIM contribution is scaled by band_w_top for rows
     * [0, band_top_rows) and band_w_rest elsewhere (mean weight over the frame
     * is 1); the divisor stays total_valid_pixels. Defaults keep it inert.
     */
    void launch_fused_ssim_mean_device(
        const float* ssim_map,
        float* temp_buffer,
        float* result_buffer,
        int N, int C, int H, int W,
        bool apply_valid_padding,
        cudaStream_t stream = nullptr,
        int band_top_rows = 0,
        float band_w_top = 1.0f,
        float band_w_rest = 1.0f);

    /**
     * @brief Reduce fused L1+SSIM loss directly to a scalar mean
     *
     * Computes mean((1-w)*abs(img1-img2) + w*(1-ssim_map)) with optional valid padding
     * without materializing an intermediate full-resolution loss map. `ssim_map`
     * is a channel-mean map with shape [N, 1, H, W].
     *
     * Round 23 diagnostic D1: trailing band_* arguments are inert unless
     * band_top_rows > 0 (see launch_fused_ssim_mean_device).
     */
    void launch_fused_l1_ssim_mean_device(
        const float* img1,
        const float* img2,
        const float* ssim_map,
        float ssim_weight,
        float* temp_buffer,
        float* result_buffer,
        int N, int C, int H, int W,
        bool apply_valid_padding,
        cudaStream_t stream = nullptr,
        int band_top_rows = 0,
        float band_w_top = 1.0f,
        float band_w_rest = 1.0f);

    void launch_fused_l1_ssim_mean_device(
        const float* img1,
        const uint8_t* img2,
        const float* ssim_map,
        float ssim_weight,
        float* temp_buffer,
        float* result_buffer,
        int N, int C, int H, int W,
        bool apply_valid_padding,
        cudaStream_t stream = nullptr,
        int band_top_rows = 0,
        float band_w_top = 1.0f,
        float band_w_rest = 1.0f);

    /**
     * @brief Reduce masked fused L1+SSIM directly to a normalized scalar loss
     *
     * Computes the masked numerator and denominator in one pass without a loss map.
     * `ssim_map` is a channel-mean map with shape [N, 1, H, W].
     * `temp_buffer` must provide room for 2 * min(1024, total_pixels/256) floats.
     */
    void launch_masked_fused_l1_ssim_mean_device(
        const float* img1,
        const float* img2,
        const float* ssim_map,
        const float* mask,
        float ssim_weight,
        float* temp_buffer,
        float* loss_buffer,
        float* mask_sum_buffer,
        int N, int C, int H, int W,
        cudaStream_t stream = nullptr);

    void launch_masked_fused_l1_ssim_mean_device(
        const float* img1,
        const float* img2,
        const float* ssim_map,
        const uint8_t* mask,
        float ssim_weight,
        float* temp_buffer,
        float* loss_buffer,
        float* mask_sum_buffer,
        int N, int C, int H, int W,
        cudaStream_t stream = nullptr);

    void launch_masked_fused_l1_ssim_mean_device(
        const float* img1,
        const uint8_t* img2,
        const float* ssim_map,
        const float* mask,
        float ssim_weight,
        float* temp_buffer,
        float* loss_buffer,
        float* mask_sum_buffer,
        int N, int C, int H, int W,
        cudaStream_t stream = nullptr);

    void launch_masked_fused_l1_ssim_mean_device(
        const float* img1,
        const uint8_t* img2,
        const float* ssim_map,
        const uint8_t* mask,
        float ssim_weight,
        float* temp_buffer,
        float* loss_buffer,
        float* mask_sum_buffer,
        int N, int C, int H, int W,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
