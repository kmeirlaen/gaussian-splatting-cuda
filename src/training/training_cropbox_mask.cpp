/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_cropbox_mask.hpp"

#include "core/scene.hpp"
#include "core/splat_data.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <span>
#include <vector>

namespace lfs::training {

    std::optional<core::Tensor> compute_cropbox_remove_mask(
        const core::Tensor& means,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        const glm::mat4& world_to_cropbox,
        const bool inverse,
        const core::Tensor* const deleted_mask) {
        if (!means.is_valid() || means.ndim() != 2 || means.size(0) == 0 || means.size(1) < 3) {
            return std::nullopt;
        }

        const std::vector<float> transform_data = {
            world_to_cropbox[0][0], world_to_cropbox[1][0], world_to_cropbox[2][0], world_to_cropbox[3][0],
            world_to_cropbox[0][1], world_to_cropbox[1][1], world_to_cropbox[2][1], world_to_cropbox[3][1],
            world_to_cropbox[0][2], world_to_cropbox[1][2], world_to_cropbox[2][2], world_to_cropbox[3][2],
            world_to_cropbox[0][3], world_to_cropbox[1][3], world_to_cropbox[2][3], world_to_cropbox[3][3]};
        auto transform_tensor = core::Tensor::from_vector(
            transform_data,
            core::TensorShape({4, 4}),
            means.device());

        const auto xyz_means = means.slice(1, 0, 3);
        auto ones = core::Tensor::ones({static_cast<size_t>(means.size(0)), 1}, means.device());
        auto means_homo = xyz_means.cat(ones, 1);

        const auto transformed_points = transform_tensor.mm(means_homo.t()).t();
        const auto local_points = transformed_points.slice(1, 0, 3);

        const std::vector<float> bbox_min_data = {crop_min.x, crop_min.y, crop_min.z};
        const std::vector<float> bbox_max_data = {crop_max.x, crop_max.y, crop_max.z};
        auto bbox_min_tensor = core::Tensor::from_vector(
            bbox_min_data, core::TensorShape({3}), means.device());
        auto bbox_max_tensor = core::Tensor::from_vector(
            bbox_max_data, core::TensorShape({3}), means.device());

        auto inside_min = local_points.ge(bbox_min_tensor.unsqueeze(0));
        auto inside_max = local_points.le(bbox_max_tensor.unsqueeze(0));
        auto inside_both = inside_min && inside_max;
        std::vector<int> reduce_dims = {1};
        auto inside_mask = inside_both.all(std::span<const int>(reduce_dims), false);
        if (!inside_mask.is_valid() || inside_mask.numel() == 0) {
            return std::nullopt;
        }

        auto remove_mask = inverse ? inside_mask : inside_mask.logical_not();
        if (deleted_mask && deleted_mask->is_valid() && deleted_mask->numel() == remove_mask.numel()) {
            remove_mask = remove_mask.logical_and(deleted_mask->logical_not());
        }

        return remove_mask;
    }

    std::optional<core::Tensor> compute_training_cropbox_remove_mask(
        const core::Scene& scene,
        const core::SplatData& model) {
        const auto training_model_name = scene.getTrainingModelNodeName();
        if (training_model_name.empty()) {
            return std::nullopt;
        }

        const auto* training_node = scene.getNode(training_model_name);
        if (!training_node) {
            return std::nullopt;
        }

        const core::NodeId cropbox_id = scene.getCropBoxForSplat(training_node->id);
        if (cropbox_id == core::NULL_NODE) {
            return std::nullopt;
        }

        const auto* cropbox = scene.getCropBoxData(cropbox_id);
        if (!cropbox || !cropbox->enabled) {
            return std::nullopt;
        }

        const auto& deleted = model.deleted();
        const core::Tensor* const deleted_mask =
            model.has_deleted_mask() && deleted.is_valid() ? &deleted : nullptr;

        return compute_cropbox_remove_mask(
            model.means(),
            cropbox->min,
            cropbox->max,
            glm::inverse(scene.getWorldTransform(cropbox_id)),
            cropbox->inverse,
            deleted_mask);
    }

} // namespace lfs::training
