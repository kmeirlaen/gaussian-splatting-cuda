/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/hdr_tonemap.hpp"
#include "io/video_frame_extractor.hpp"
#include "io/video/video_encoder.hpp"

extern "C" {
#include <libavutil/pixfmt.h>
}

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

    using lfs::io::ExtractionMode;
    using lfs::io::HdrFormat;
    using lfs::io::ResolutionMode;
    using lfs::io::VideoFrameExtractor;

    struct TempDir {
        explicit TempDir(const std::string_view label) {
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            path = std::filesystem::temp_directory_path() /
                   ("lfs_video_extract_" + std::string(label) + "_" + std::to_string(now));
            std::filesystem::create_directories(path);
        }

        ~TempDir() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }

        std::filesystem::path path;
    };

    struct CudaFloatBuffer {
        explicit CudaFloatBuffer(const std::size_t count) {
            status = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float));
        }

        ~CudaFloatBuffer() {
            if (ptr)
                cudaFree(ptr);
        }

        float* ptr = nullptr;
        cudaError_t status = cudaSuccess;
    };

    bool writeTinyEncodedVideo(const std::filesystem::path& video_path, std::string& error) {
        constexpr int width = 16;
        constexpr int height = 16;
        constexpr int frame_count = 5;
        constexpr int channels = 3;

        lfs::io::video::VideoExportOptions options;
        options.preset = lfs::io::video::VideoPreset::CUSTOM;
        options.width = width;
        options.height = height;
        options.framerate = 10;
        options.crf = 23;

        lfs::io::video::VideoEncoder encoder;
        if (const auto opened = encoder.open(video_path, options); !opened) {
            error = opened.error();
            return false;
        }

        std::vector<float> frame(static_cast<std::size_t>(width) * height * channels);
        CudaFloatBuffer device_frame(frame.size());
        if (device_frame.status != cudaSuccess) {
            error = cudaGetErrorString(device_frame.status);
            return false;
        }

        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            const float red = static_cast<float>(frame_index + 1) / static_cast<float>(frame_count);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const std::size_t offset = (static_cast<std::size_t>(y) * width + x) * channels;
                    frame[offset + 0] = red;
                    frame[offset + 1] = static_cast<float>(x) / static_cast<float>(width - 1);
                    frame[offset + 2] = static_cast<float>(y) / static_cast<float>(height - 1);
                }
            }

            const cudaError_t copy_status = cudaMemcpy(
                device_frame.ptr, frame.data(), frame.size() * sizeof(float), cudaMemcpyHostToDevice);
            if (copy_status != cudaSuccess) {
                error = cudaGetErrorString(copy_status);
                return false;
            }

            if (const auto written = encoder.writeFrameGpu(device_frame.ptr, width, height); !written) {
                error = written.error();
                return false;
            }
        }

        if (const auto closed = encoder.close(); !closed) {
            error = closed.error();
            return false;
        }
        return true;
    }

    VideoFrameExtractor::Params extractionParams(
        const std::filesystem::path& video_path,
        const std::filesystem::path& output_dir) {
        VideoFrameExtractor::Params params;
        params.video_path = video_path;
        params.output_dir = output_dir;
        params.mode = ExtractionMode::INTERVAL;
        params.frame_interval = 1;
        params.format = lfs::io::ImageFormat::PNG;
        params.generate_metadata = true;
        return params;
    }
    VideoFrameExtractor::Params validParams() {
        VideoFrameExtractor::Params params;
        params.mode = ExtractionMode::FPS;
        params.fps = 1.0;
        params.start_time = 0.0;
        params.end_time = 10.0;
        return params;
    }

} // namespace

TEST(SparseExtractionContract, TargetCountUsesHalfOpenTrimRange) {
    EXPECT_EQ(lfs::io::calculateFpsSampleCount(0.0, 10.0, 1.0), 10u);
    EXPECT_EQ(lfs::io::calculateFpsSampleCount(2.25, 2.26, 1.0), 1u);
    EXPECT_EQ(lfs::io::calculateFpsSampleCount(0.0, 1.0001, 2.0), 3u);
    EXPECT_EQ(lfs::io::calculateFpsSampleCount(0.0, 1.0, 0.0), 0u);

    const std::size_t target_count =
        lfs::io::calculateFpsSampleCount(2.0, 10.0, 1.0);
    ASSERT_EQ(target_count, 8u);
    EXPECT_DOUBLE_EQ(
        lfs::io::fpsSampleTime(2.0, 10.0, 1.0, target_count - 1), 9.0);
    EXPECT_LT(lfs::io::fpsSampleTime(2.0, 10.0, 1.0, 8), 10.0);
}

TEST(SparseExtractionContract, FinalFrameMaySatisfyCoveredTailTarget) {
    EXPECT_TRUE(lfs::io::frameCoversSampleTime(9.0, 1.0, 9.75));
    EXPECT_FALSE(lfs::io::frameCoversSampleTime(9.0, 1.0, 10.0));
    EXPECT_FALSE(lfs::io::frameCoversSampleTime(9.0, 0.5, 9.75));
}

TEST(SparseExtractionContract, PastEndFillsCoveredRetainedTail) {
    constexpr double start_time = 0.0;
    constexpr double end_time = 10.0;
    constexpr double target_fps = 1.0;
    constexpr double frame_duration = 2.0;
    constexpr std::array frame_times{0.0, 2.0, 4.0, 8.0};
    constexpr double next_decoded_frame_time = 10.5;

    const std::size_t target_count =
        lfs::io::calculateFpsSampleCount(start_time, end_time, target_fps);
    std::size_t next_target_index = 8;
    const bool reached_eof = false;
    const bool reached_end = next_decoded_frame_time >= end_time;

    ASSERT_EQ(target_count, 10u);
    EXPECT_TRUE(lfs::io::frameCoversSampleTime(
        frame_times.back(), frame_duration,
        lfs::io::fpsSampleTime(start_time, end_time, target_fps, 8)));
    EXPECT_TRUE(lfs::io::frameCoversSampleTime(
        frame_times.back(), frame_duration,
        lfs::io::fpsSampleTime(start_time, end_time, target_fps, 9)));
    ASSERT_TRUE(lfs::io::shouldFillRetainedFpsTail(reached_eof, reached_end));
    while (lfs::io::shouldFillRetainedFpsTail(reached_eof, reached_end) &&
           next_target_index < target_count) {
        const double target_time = lfs::io::fpsSampleTime(
            start_time, end_time, target_fps, next_target_index);
        if (!lfs::io::frameCoversSampleTime(
                frame_times.back(), frame_duration, target_time)) {
            break;
        }
        ++next_target_index;
    }

    EXPECT_EQ(next_target_index, target_count);
}

TEST(HdrFormatClassification, NativeProfileFivePrecedesCompatibilityMetadata) {
    EXPECT_EQ(
        lfs::io::detectDolbyVisionFormat(AVCOL_TRC_SMPTE2084, 5, 1),
        HdrFormat::DOLBY_VISION_NATIVE);
    EXPECT_EQ(
        lfs::io::detectDolbyVisionFormat(AVCOL_TRC_SMPTE2084, 8, 1),
        HdrFormat::DOLBY_VISION_HDR10);
    EXPECT_EQ(
        lfs::io::detectDolbyVisionFormat(AVCOL_TRC_UNSPECIFIED, 8, 4),
        HdrFormat::DOLBY_VISION_HLG);
    EXPECT_EQ(
        lfs::io::detectDolbyVisionFormat(AVCOL_TRC_SMPTE2084, 7, 1),
        HdrFormat::DOLBY_VISION_NATIVE);
}

TEST(HdrFormatClassification, StaticHdrMetadataRequiresTenBitSource) {
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_UNSPECIFIED, 10, true, false),
        HdrFormat::HDR10);
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_UNSPECIFIED, 10, false, true),
        HdrFormat::HDR10);
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_UNSPECIFIED, 8, true, true),
        HdrFormat::SDR);
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_UNSPECIFIED, 10, false, false),
        HdrFormat::SDR);
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_SMPTE2084, 8),
        HdrFormat::HDR10);
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_ARIB_STD_B67, 8),
        HdrFormat::HLG);
}

TEST(VideoFrameExtractorParams, ComputesCheckedOutputLayout) {
    auto params = validParams();
    params.resolution_mode = ResolutionMode::Scale;
    params.scale = 0.5f;

    VideoFrameExtractor::ValidatedLayout layout;
    std::string error;
    ASSERT_TRUE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error))
        << error;
    EXPECT_EQ(layout.width, 960);
    EXPECT_EQ(layout.height, 540);
    EXPECT_EQ(layout.rgb_bytes, 960u * 540u * 3u);
}

TEST(VideoFrameExtractorParams, RejectsInvalidRangesAndDimensions) {
    VideoFrameExtractor::ValidatedLayout layout;
    std::string error;

    auto params = validParams();
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 0, 1080, 1.0 / 90000.0, layout, error));
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 0.0, layout, error));

    params.start_time = 10.0;
    params.end_time = 10.0;
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));

    params = validParams();
    params.rotation = 45;
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));

    params = validParams();
    params.fps = std::numeric_limits<double>::max();
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));

    params = validParams();
    params.end_time = 1.0e9;
    params.fps = 3.0;
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));

    params = validParams();
    params.resolution_mode = ResolutionMode::Scale;
    params.scale = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));

    params = validParams();
    params.resolution_mode = ResolutionMode::Custom;
    params.custom_width = std::numeric_limits<int>::max() / 2;
    params.custom_height = 1;
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));
}

TEST(VideoFrameExtractorOutputNaming, IntervalUsesSourceFrameNumbers) {
    TempDir temp("interval");
    const std::filesystem::path video_path = temp.path / "source.mp4";
    const std::filesystem::path output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);

    std::string error;
    ASSERT_TRUE(writeTinyEncodedVideo(video_path, error)) << error;

    auto params = extractionParams(video_path, output_dir);
    params.frame_interval = 2;

    VideoFrameExtractor extractor;
    EXPECT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_1.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_3.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_5.png"));
    EXPECT_FALSE(std::filesystem::exists(output_dir / "frame_2.png"));
}

TEST(VideoFrameExtractorOutputNaming, TrimmedRangeKeepsOriginalSourceFrameNumbers) {
    TempDir temp("trim");
    const std::filesystem::path video_path = temp.path / "source.mp4";
    const std::filesystem::path output_dir = temp.path / "frames";
    std::filesystem::create_directories(output_dir);

    std::string error;
    ASSERT_TRUE(writeTinyEncodedVideo(video_path, error)) << error;

    auto params = extractionParams(video_path, output_dir);
    params.start_time = 0.19;
    params.end_time = 0.31;

    VideoFrameExtractor extractor;
    EXPECT_TRUE(extractor.extract(params, error)) << error;
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_3.png"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "frame_4.png"));
    EXPECT_FALSE(std::filesystem::exists(output_dir / "frame_1.png"));
}
