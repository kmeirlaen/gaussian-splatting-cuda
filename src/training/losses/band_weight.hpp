/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Round 23 diagnostic D1: row-band weighting of the photometric loss.
//
// Gate LFS_EXP_BAND_WEIGHT=<factor> (e.g. 4.0): the per-pixel photometric
// loss and its gradient (L1 and DSSIM parts alike) are multiplied by
// <factor> for pixel rows in the top third of the image and by 1
// elsewhere, then renormalized (w / mean(w)) so the mean weight over the
// frame is exactly 1. This is a diagnostic of objective expressiveness for
// the far band, NOT a feature: default off, bit-identical when unset, no
// user-facing parameter.
//
// Weights depend only on the pixel row, so two scalars plus the row count
// fully describe the map. When the gate is unset (or degenerate) `active`
// is false and every consumer must skip the weighting branch entirely so
// the executed arithmetic stays bit-identical to the unpatched kernel.

#include <cmath>
#include <cstdlib>

namespace lfs::training::losses {

    struct BandWeightSpec {
        bool active = false;
        float factor = 1.0f; // raw <factor> from the environment
        int top_rows = 0;    // rows [0, top_rows) receive w_top
        float w_top = 1.0f;  // normalized weights (mean over frame == 1)
        float w_rest = 1.0f;
    };

    // Parses LFS_EXP_BAND_WEIGHT once per process (cached static).
    inline BandWeightSpec make_band_weight_spec(const int height) {
        BandWeightSpec spec;

        static const float env_factor = [] {
            const char* s = std::getenv("LFS_EXP_BAND_WEIGHT");
            if (!s || !*s)
                return 0.0f;
            char* end = nullptr;
            const float v = std::strtof(s, &end);
            if (end == s || !std::isfinite(v) || v <= 0.0f)
                return 0.0f;
            return v;
        }();

        if (env_factor <= 0.0f || height <= 0)
            return spec;

        const int top_rows = height / 3;
        if (top_rows <= 0 || top_rows >= height)
            return spec; // degenerate banding on tiny images: stay inert

        const double mean_w =
            (static_cast<double>(env_factor) * top_rows + (height - top_rows)) / height;

        spec.active = true;
        spec.factor = env_factor;
        spec.top_rows = top_rows;
        spec.w_top = static_cast<float>(env_factor / mean_w);
        spec.w_rest = static_cast<float>(1.0 / mean_w);
        return spec;
    }

} // namespace lfs::training::losses
