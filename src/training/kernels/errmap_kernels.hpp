/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // All launchers write [H,W] float32 maps and fall back to the calling thread's
    // current CUDA stream when no stream is passed (kernel_stream.hpp).

    // Mode 1: per-pixel mean over channels of |render - gt| (both CHW float32).
    void launch_errmap_l1_residual(
        const float* d_render_chw,
        const float* d_gt_chw,
        int channels,
        int height,
        int width,
        float* d_out_hw,
        cudaStream_t stream = nullptr);

    // Rec601 luma (0.299 R + 0.587 G + 0.114 B) of a CHW float32 image -> [H,W].
    // A single-channel input is copied through unchanged.
    void launch_errmap_luma_rec601(
        const float* d_img_chw,
        int channels,
        int height,
        int width,
        float* d_luma_hw,
        cudaStream_t stream = nullptr);

    // One pass of the separable 7-tap Gaussian (sigma = 1.5, edge-clamped).
    // horizontal=true sweeps along x, otherwise along y. In-place unsafe.
    void launch_errmap_gauss7_pass(
        const float* d_in_hw,
        int height,
        int width,
        bool horizontal,
        float* d_out_hw,
        cudaStream_t stream = nullptr);

    // detail_deficit = max(0, HF(gt) - HF(render)) with HF(x) = |x - blur(x)|.
    void launch_errmap_detail_deficit(
        const float* d_gt_hw,
        const float* d_gt_blurred_hw,
        const float* d_render_hw,
        const float* d_render_blurred_hw,
        std::size_t n,
        float* d_out_hw,
        cudaStream_t stream = nullptr);

    // Blend of two already mean-normalized maps: out = 0.5 * a + 0.5 * b.
    void launch_errmap_mix(
        const float* d_a_hw,
        const float* d_b_hw,
        std::size_t n,
        float* d_out_hw,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
