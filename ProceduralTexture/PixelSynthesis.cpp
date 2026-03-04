#include "PixelSynthesis.h"
#include <random>
#include <iostream>

namespace EBPTns {

    PixelSynthesis::PixelSynthesis() {
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }

    void PixelSynthesis::setRandomSeed(unsigned int seed) {
        rng_.seed(seed);
    }

    cv::Rect PixelSynthesis::getSourcePatchRect(const cv::Rect& target_bbox,
        const cv::Mat& input_image,
        const EdgeGroup* source_group) {
        if (source_group != nullptr) {
            cv::Rect source_bbox = source_group->getBoundingBox();
            if (source_bbox.x < 0) source_bbox.x = 0;
            if (source_bbox.y < 0) source_bbox.y = 0;
            if (source_bbox.x + source_bbox.width > input_image.cols)
                source_bbox.width = input_image.cols - source_bbox.x;
            if (source_bbox.y + source_bbox.height > input_image.rows)
                source_bbox.height = input_image.rows - source_bbox.y;
            if (source_bbox.width <= 0 || source_bbox.height <= 0) {
                return getSourcePatchRect(target_bbox, input_image, nullptr);
            }
            return source_bbox;
        }
        if (input_image.empty() || target_bbox.width <= 0 || target_bbox.height <= 0) {
            return cv::Rect(0, 0, 0, 0);
        }
        if (target_bbox.width > input_image.cols || target_bbox.height > input_image.rows) {
            return cv::Rect(0, 0, input_image.cols, input_image.rows);
        }
        if (patch_method_ == 1 && source_group != nullptr) {
            cv::Point2f center = source_group->getCenter();
            int center_x = static_cast<int>(center.x) % input_image.cols;
            int center_y = static_cast<int>(center.y) % input_image.rows;
            int patch_x = center_x - target_bbox.width / 2;
            int patch_y = center_y - target_bbox.height / 2;
            patch_x = std::max(0, std::min(patch_x, input_image.cols - target_bbox.width));
            patch_y = std::max(0, std::min(patch_y, input_image.rows - target_bbox.height));
            return cv::Rect(patch_x, patch_y, target_bbox.width, target_bbox.height);
        }
        else {
            int max_x = input_image.cols - target_bbox.width;
            int max_y = input_image.rows - target_bbox.height;
            if (max_x <= 0 || max_y <= 0) {
                return cv::Rect(0, 0,
                    std::min(target_bbox.width, input_image.cols),
                    std::min(target_bbox.height, input_image.rows));
            }
            std::uniform_int_distribution<int> dist_x(0, max_x);
            std::uniform_int_distribution<int> dist_y(0, max_y);
            return cv::Rect(dist_x(rng_), dist_y(rng_),
                target_bbox.width, target_bbox.height);
        }
    }

    void PixelSynthesis::blendPatches(cv::Mat& output, const cv::Mat& patch,
        const cv::Rect& bbox, float alpha) {
        if (patch.empty() || output.empty() ||
            bbox.width <= 0 || bbox.height <= 0 ||
            bbox.x < 0 || bbox.y < 0 ||
            bbox.x + bbox.width > output.cols ||
            bbox.y + bbox.height > output.rows) {
            return;
        }
        if (alpha >= 0.99f) {
            patch.copyTo(output(bbox));
        }
        else {
            cv::Mat roi = output(bbox);
            cv::addWeighted(patch, alpha, roi, 1.0f - alpha, 0, roi);
        }
    }

    /*cv::Mat PixelSynthesis::PatchCopy(
        const cv::Mat& input_image,
        const std::vector<EdgeGroup>& source_groups,
        const std::vector<PlacedGroup>& placed_groups,
        int output_width, int output_height) {

        if (input_image.empty() || placed_groups.empty() || source_groups.empty()) {
            return cv::Mat::zeros(output_height, output_width, input_image.type());
        }
        cv::Mat output = cv::Mat::zeros(output_height, output_width, input_image.type());
        int processed_groups = 0;
        int skipped_groups = 0;
        for (const auto& placed_group : placed_groups) {
            const auto& group = placed_group.group;
            if (placed_group.source_index < 0 ||
                placed_group.source_index >= static_cast<int>(source_groups.size())) {
                skipped_groups++;
                continue;
            }
            cv::Rect bbox = group.getBoundingBox();
            if (bbox.x < 0) bbox.x = 0;
            if (bbox.y < 0) bbox.y = 0;
            if (bbox.x + bbox.width > output_width) bbox.width = output_width - bbox.x;
            if (bbox.y + bbox.height > output_height) bbox.height = output_height - bbox.y;
            if (bbox.width <= 0 || bbox.height <= 0) {
                skipped_groups++;
                continue;
            }
            const EdgeGroup& source_group = source_groups[placed_group.source_index];
            cv::Rect source_bbox = source_group.getBoundingBox();
            if (source_bbox.width <= 0 || source_bbox.height <= 0 ||
                source_bbox.x < 0 || source_bbox.y < 0 ||
                source_bbox.x + source_bbox.width > input_image.cols ||
                source_bbox.y + source_bbox.height > input_image.rows) {
                skipped_groups++;
                continue;
            }
            cv::Mat source_patch = input_image(source_bbox).clone();
            cv::Mat resized_patch;
            cv::resize(source_patch, resized_patch, bbox.size());
            blendPatches(output, resized_patch, bbox, 1.0f);
            processed_groups++;
        }
        if (processed_groups == 0) {
            cv::Scalar avg_color = cv::mean(input_image);
            output = cv::Mat(output_height, output_width, input_image.type(), avg_color);
            cv::Mat noise = cv::Mat(output_height, output_width, input_image.type());
            cv::randn(noise, 0, 25);
            output += noise;
        }
        return output;
    }

    cv::Mat PixelSynthesis::fillPixels(
        const cv::Mat& input_image,
        const std::vector<EdgeGroup>& source_groups,
        const std::vector<PlacedGroup>& placed_groups,
        int output_width, int output_height) {

        return PatchCopy(input_image, source_groups, placed_groups,
            output_width, output_height);
    }*/

    cv::Mat PixelSynthesis::copyWithMask(const cv::Mat& source, const cv::Mat& mask,
        const cv::Rect& source_bbox, const cv::Rect& target_bbox) {
        cv::Mat result = cv::Mat::zeros(target_bbox.size(), source.type());

        for (int y = 0; y < target_bbox.height; ++y) {
            for (int x = 0; x < target_bbox.width; ++x) {
                int source_x = source_bbox.x + x;
                int source_y = source_bbox.y + y;

                if (source_x >= 0 && source_x < source.cols &&
                    source_y >= 0 && source_y < source.rows) {

                    if (mask.empty() || mask.at<uchar>(source_y, source_x) > 0) {
                        result.at<cv::Vec3b>(y, x) = source.at<cv::Vec3b>(source_y, source_x);
                    }
                }
            }
        }

        return result;
    }

    cv::Mat PixelSynthesis::PatchCopy(
        const cv::Mat& input_image,
        const std::vector<SourceGroupInfo>& source_infos,
        const std::vector<PlacedGroup>& placed_groups,
        int output_width, int output_height) {

        cv::Mat output = cv::Mat::zeros(output_height, output_width, input_image.type());

        for (const auto& placed : placed_groups) {
            if (placed.source_index < 0 || placed.source_index >= source_infos.size()) {
                continue;
            }

            const auto& source_info = source_infos[placed.source_index];
            cv::Rect source_bbox = source_info.group.getBoundingBox();
            cv::Rect target_bbox = placed.group.getBoundingBox();

            if (target_bbox.x < 0) target_bbox.x = 0;
            if (target_bbox.y < 0) target_bbox.y = 0;
            if (target_bbox.x + target_bbox.width > output_width)
                target_bbox.width = output_width - target_bbox.x;
            if (target_bbox.y + target_bbox.height > output_height)
                target_bbox.height = output_height - target_bbox.y;

            if (target_bbox.width <= 0 || target_bbox.height <= 0) continue;

            cv::Mat mask = source_info.superpixel_mask;

            cv::Mat patch = copyWithMask(input_image, mask, source_bbox, target_bbox);

            // Масштабируем если нужно
            if (patch.cols != target_bbox.width || patch.rows != target_bbox.height) {
                cv::resize(patch, patch, target_bbox.size());
            }

            patch.copyTo(output(target_bbox));
        }

        return output;
    }

}