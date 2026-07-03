/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/formats/colmap.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

namespace {
    std::filesystem::path write_points3d_fixture(const std::string& name, const std::string& contents) {
        const auto dir = std::filesystem::temp_directory_path() / "lfs_colmap_points3d_text_tests";
        std::filesystem::create_directories(dir);
        const auto path = dir / name;
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << contents;
        return path;
    }

    void expect_point(const lfs::io::detail::ColmapPoint3DTextPointCloudData& data,
                      const size_t index,
                      const float x,
                      const float y,
                      const float z,
                      const uint8_t r,
                      const uint8_t g,
                      const uint8_t b) {
        ASSERT_LT(index * 3 + 2, data.positions.size());
        ASSERT_LT(index * 3 + 2, data.colors.size());
        EXPECT_FLOAT_EQ(data.positions[index * 3 + 0], x);
        EXPECT_FLOAT_EQ(data.positions[index * 3 + 1], y);
        EXPECT_FLOAT_EQ(data.positions[index * 3 + 2], z);
        EXPECT_EQ(data.colors[index * 3 + 0], r);
        EXPECT_EQ(data.colors[index * 3 + 1], g);
        EXPECT_EQ(data.colors[index * 3 + 2], b);
    }
} // namespace

TEST(ColmapPoints3DText, FastPathParsesBasicValuesAndIgnoresTracks) {
    const auto path = write_points3d_fixture(
        "basic_points3D.txt",
        "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
        "1 1.5 -2.25 3.75 10 20 30 0.125 7 8 9 10\n"
        "2 -4 5 6.5 255 128 0 2.0 11 12\n");

    const auto data = lfs::io::detail::parse_points3D_text_point_cloud_fast(path);
    EXPECT_EQ(data.point_count, 2u);
    EXPECT_EQ(data.file_lines, 3u);
    expect_point(data, 0, 1.5f, -2.25f, 3.75f, 10, 20, 30);
    expect_point(data, 1, -4.0f, 5.0f, 6.5f, 255, 128, 0);
}

TEST(ColmapPoints3DText, FastPathSkipsCommentsBlanksAndHandlesCrLf) {
    const auto path = write_points3d_fixture(
        "crlf_points3D.txt",
        "# header\r\n"
        "\r\n"
        "3 0 1 2 1 2 3 0.0 4 5\r\n"
        "\r\n"
        "# footer\r\n");

    const auto data = lfs::io::detail::parse_points3D_text_point_cloud_fast(path);
    EXPECT_EQ(data.point_count, 1u);
    EXPECT_EQ(data.file_lines, 5u);
    expect_point(data, 0, 0.0f, 1.0f, 2.0f, 1, 2, 3);
}

TEST(ColmapPoints3DText, FastPathHandlesFinalLineWithoutTrailingNewline) {
    const auto path = write_points3d_fixture(
        "no_trailing_newline_points3D.txt",
        "4 9 8 7 6 5 4 0.0");

    const auto data = lfs::io::detail::parse_points3D_text_point_cloud_fast(path);
    EXPECT_EQ(data.point_count, 1u);
    EXPECT_EQ(data.file_lines, 1u);
    expect_point(data, 0, 9.0f, 8.0f, 7.0f, 6, 5, 4);
}

TEST(ColmapPoints3DText, FastPathDoesNotParseLongTrackTails) {
    std::string line = "5 1 2 3 4 5 6 0.0";
    for (int i = 0; i < 500; ++i) {
        line += " " + std::to_string(i) + " " + std::to_string(i + 1);
    }
    line += "\n";

    const auto path = write_points3d_fixture("long_track_tail_points3D.txt", line);
    const auto data = lfs::io::detail::parse_points3D_text_point_cloud_fast(path);
    EXPECT_EQ(data.point_count, 1u);
    expect_point(data, 0, 1.0f, 2.0f, 3.0f, 4, 5, 6);
}

TEST(ColmapPoints3DText, FastPathThrowsOnMalformedLine) {
    const auto path = write_points3d_fixture(
        "malformed_points3D.txt",
        "6 1 2 3 4 5 6\n");

    EXPECT_THROW(
        (void)lfs::io::detail::parse_points3D_text_point_cloud_fast(path),
        std::runtime_error);
}

TEST(ColmapPoints3DText, FastPathThrowsOnEmptyOrCommentOnlyFile) {
    const auto path = write_points3d_fixture(
        "empty_points3D.txt",
        "# only comments\n\n# still no points\n");

    EXPECT_THROW(
        (void)lfs::io::detail::parse_points3D_text_point_cloud_fast(path),
        std::runtime_error);
}

TEST(ColmapPoints3DText, RecordFullModeParsesTracksExactly) {
    const auto path = write_points3d_fixture(
        "full_tracks_points3D.txt",
        "10 1 2 3 7 8 9 0.25 100 200 101 201\n");

    const auto records = lfs::io::detail::parse_points3D_text_records_for_test(
        path,
        lfs::io::detail::ColmapPoint3DTrackParseMode::Full);
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].point3D_id, 10u);
    EXPECT_DOUBLE_EQ(records[0].xyz[0], 1.0);
    EXPECT_DOUBLE_EQ(records[0].xyz[1], 2.0);
    EXPECT_DOUBLE_EQ(records[0].xyz[2], 3.0);
    EXPECT_EQ(records[0].color[0], 7);
    EXPECT_EQ(records[0].color[1], 8);
    EXPECT_EQ(records[0].color[2], 9);
    EXPECT_DOUBLE_EQ(records[0].error, 0.25);
    EXPECT_EQ(records[0].track_count, 2u);
    ASSERT_EQ(records[0].track.size(), 2u);
    EXPECT_EQ(records[0].track[0].image_id, 100u);
    EXPECT_EQ(records[0].track[0].point2D_idx, 200u);
    EXPECT_EQ(records[0].track[1].image_id, 101u);
    EXPECT_EQ(records[0].track[1].point2D_idx, 201u);
}

TEST(ColmapPoints3DText, RecordCountOnlyMatchesFullTrackCounts) {
    const auto path = write_points3d_fixture(
        "count_only_tracks_points3D.txt",
        "11 0 0 0 1 2 3 0.0 1 2 3 4 5 6\n"
        "12 0 0 0 4 5 6 0.0 7 8 9\n");

    const auto full = lfs::io::detail::parse_points3D_text_records_for_test(
        path,
        lfs::io::detail::ColmapPoint3DTrackParseMode::Full);
    const auto count_only = lfs::io::detail::parse_points3D_text_records_for_test(
        path,
        lfs::io::detail::ColmapPoint3DTrackParseMode::CountOnly);

    ASSERT_EQ(full.size(), count_only.size());
    EXPECT_EQ(full[0].track_count, 3u);
    EXPECT_EQ(count_only[0].track_count, full[0].track_count);
    EXPECT_TRUE(count_only[0].track.empty());
    EXPECT_EQ(full[1].track_count, 1u);
    EXPECT_EQ(count_only[1].track_count, full[1].track_count);
    EXPECT_TRUE(count_only[1].track.empty());
}

TEST(ColmapPoints3DText, RecordNoneModeLeavesTrackCountZero) {
    const auto path = write_points3d_fixture(
        "none_tracks_points3D.txt",
        "13 0 0 0 1 2 3 0.0 1 2 3 4\n");

    const auto records = lfs::io::detail::parse_points3D_text_records_for_test(
        path,
        lfs::io::detail::ColmapPoint3DTrackParseMode::None);
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].track_count, 0u);
    EXPECT_TRUE(records[0].track.empty());
}

TEST(ColmapPoints3DText, RecordPathThrowsOnMalformedHeader) {
    const auto path = write_points3d_fixture(
        "malformed_record_points3D.txt",
        "14 0 0 0 1 2 3\n");

    EXPECT_THROW(
        (void)lfs::io::detail::parse_points3D_text_records_for_test(
            path,
            lfs::io::detail::ColmapPoint3DTrackParseMode::CountOnly),
        std::runtime_error);
}

TEST(ColmapPoints3DText, LargeFileParallelPathIsDeterministicAcrossModes) {
    std::ostringstream contents;
    contents << "# generated large fixture\n";
    constexpr int kPointCount = 320000;
    for (int i = 0; i < kPointCount; ++i) {
        contents << i << ' '
                 << static_cast<double>(i) * 0.5 << ' '
                 << static_cast<double>(i) * -0.25 << ' '
                 << static_cast<double>(i) * 0.125 << ' '
                 << (i % 256) << ' '
                 << ((i + 1) % 256) << ' '
                 << ((i + 2) % 256) << " 0.0 "
                 << i << ' ' << (i + 10) << ' '
                 << (i + 20) << ' ' << (i + 30) << '\n';
    }

    const auto path = write_points3d_fixture("large_parallel_points3D.txt", contents.str());
    const auto fast = lfs::io::detail::parse_points3D_text_point_cloud_fast(path);
    const auto full = lfs::io::detail::parse_points3D_text_records_for_test(
        path,
        lfs::io::detail::ColmapPoint3DTrackParseMode::Full);
    const auto count_only = lfs::io::detail::parse_points3D_text_records_for_test(
        path,
        lfs::io::detail::ColmapPoint3DTrackParseMode::CountOnly);

    ASSERT_EQ(fast.point_count, static_cast<size_t>(kPointCount));
    ASSERT_EQ(full.size(), static_cast<size_t>(kPointCount));
    ASSERT_EQ(count_only.size(), static_cast<size_t>(kPointCount));
    expect_point(fast, 0, 0.0f, -0.0f, 0.0f, 0, 1, 2);
    expect_point(fast, kPointCount - 1,
                 static_cast<float>(kPointCount - 1) * 0.5f,
                 static_cast<float>(kPointCount - 1) * -0.25f,
                 static_cast<float>(kPointCount - 1) * 0.125f,
                 static_cast<uint8_t>((kPointCount - 1) % 256),
                 static_cast<uint8_t>(kPointCount % 256),
                 static_cast<uint8_t>((kPointCount + 1) % 256));
    EXPECT_EQ(full.front().track_count, 2u);
    EXPECT_EQ(full.back().track_count, 2u);
    EXPECT_EQ(count_only.front().track_count, full.front().track_count);
    EXPECT_EQ(count_only.back().track_count, full.back().track_count);
    EXPECT_TRUE(count_only.front().track.empty());
    EXPECT_TRUE(count_only.back().track.empty());
}
