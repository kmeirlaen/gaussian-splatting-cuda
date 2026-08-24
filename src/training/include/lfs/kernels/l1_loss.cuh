/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Fused L1 loss computation with gradient
     *
     * Computes in a single optimized pass:
     * - loss = mean(|img1 - img2|)
     * - grad = sign(img1 - img2) / N
     *
     * @param img1 Input image 1 (N elements)
     * @param img2 Input image 2 (N elements)
     * @param grad_out Output gradient (N elements)
     * @param loss_out Output scalar loss (1 element)
     * @param temp_buffer Temporary buffer for partial sums (min(1024, (N+255)/256) elements)
     * @param N Number of elements
     * @param stream CUDA stream
     *
     * Round 23 diagnostic D1 (LFS_EXP_BAND_WEIGHT): when band_top_rows > 0,
     * both the per-pixel |diff| contribution and the gradient are scaled by
     * band_w_top for rows [0, band_top_rows) and band_w_rest elsewhere
     * (weights pre-normalized so the mean over the frame is 1). Default
     * arguments keep the gate off and the arithmetic unchanged.
     */
    void launch_fused_l1_loss(
        const float* img1,
        const float* img2,
        float* grad_out,
        float* loss_out,
        float* temp_buffer,
        size_t N,
        cudaStream_t stream = nullptr,
        int band_image_h = 0,
        int band_image_w = 0,
        int band_top_rows = 0,
        float band_w_top = 1.0f,
        float band_w_rest = 1.0f);

    void launch_fused_l1_loss(
        const float* img1,
        const uint8_t* img2,
        float* grad_out,
        float* loss_out,
        float* temp_buffer,
        size_t N,
        cudaStream_t stream = nullptr,
        int band_image_h = 0,
        int band_image_w = 0,
        int band_top_rows = 0,
        float band_w_top = 1.0f,
        float band_w_rest = 1.0f);

} // namespace lfs::training::kernels
