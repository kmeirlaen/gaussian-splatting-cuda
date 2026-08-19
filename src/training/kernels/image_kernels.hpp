/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_fused_canny_edge_filter_chw(
        const float* d_input_chw,
        float* d_output_hw,
        const int height,
        const int width,
        cudaStream_t stream = nullptr);
    void launch_fused_canny_edge_filter_chw(
        const uint8_t* d_input_chw,
        float* d_output_hw,
        const int height,
        const int width,
        cudaStream_t stream = nullptr);
    // Divides d_data by *d_scalar in place; the whole kernel no-ops when
    // *d_scalar <= skip_below (device-side replacement for host mean/median
    // readback + branch).
    void launch_normalize_by_device_scalar(
        float* d_data,
        const std::size_t n,
        const float* d_scalar,
        float skip_below = 0.0f,
        cudaStream_t stream = nullptr);

    /**
     * Add a Difference-of-Gaussians residual to a densification error map:
     *   dog(x) = |G(x, sigma=1) - G(x, sigma=2)|  on luminance
     *   error += weight * |dog(gt) - dog(render)|
     *
     * Images are contiguous RGB, CHW ([3,H,W] or [1,3,H,W]) or HWC ([H,W,3]).
     * GT may be float32 or uint8; render is float32. scratch must hold 3*H*W floats.
     */
    void launch_add_dog_residual(
        const float* gt_f32,
        const std::uint8_t* gt_u8,
        bool gt_chw,
        const float* render_f32,
        bool render_chw,
        float* error_hw,
        int height,
        int width,
        float weight,
        float* scratch_3hw,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
