#include "TextureSynthesis.h"
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace EBPTns {

    TextureSynthesis::TextureSynthesis() {
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }

    void TextureSynthesis::setRandomSeed(unsigned int seed) {
        rng_.seed(seed);
    }

    cv::Point2f TextureSynthesis::generateRandomPosition(int width, int height) {
        if (width <= 0 || height <= 0) {
            return cv::Point2f(0, 0);
        }
        float margin = 50.0f;
        std::uniform_real_distribution<float> dist_x(
            margin,
            std::max(margin, static_cast<float>(width) - margin)
        );
        std::uniform_real_distribution<float> dist_y(
            margin,
            std::max(margin, static_cast<float>(height) - margin)
        );
        return cv::Point2f(dist_x(rng_), dist_y(rng_));
    }

    float TextureSynthesis::generateRandomAngle(float base_angle, float variation) {
        if (variation <= 0.0f) {
            return base_angle;
        }
        float max_variation = variation * CV_PI;
        std::uniform_real_distribution<float> dist(-max_variation, max_variation);
        float random_angle = base_angle + dist(rng_);
        while (random_angle < 0) random_angle += CV_PI;
        while (random_angle >= CV_PI) random_angle -= CV_PI;
        return random_angle;
    }

    float TextureSynthesis::generateRandomScale(float base_scale, float variation) {
        if (variation <= 0.0f) {
            return base_scale;
        }
        float min_scale = base_scale * (1.0f - variation);
        float max_scale = base_scale * (1.0f + variation);
        std::uniform_real_distribution<float> dist(min_scale, max_scale);
        return std::max(0.1f, dist(rng_));
    }

    bool TextureSynthesis::checkOverlap(const EdgeGroup& group1,
        const EdgeGroup& group2,
        float min_distance) {
        if (group1.getEdges().empty() || group2.getEdges().empty()) {
            return false;
        }
        cv::Point2f center1 = group1.getCenter();
        cv::Point2f center2 = group2.getCenter();
        float dx = center1.x - center2.x;
        float dy = center1.y - center2.y;
        float distance = std::sqrt(dx * dx + dy * dy);
        float combined_radius = group1.getRadialSpread() + group2.getRadialSpread();
        return distance < (combined_radius + min_distance);
    }

    PlacedGroup TextureSynthesis::transformGroup(const EdgeGroup& source_group,
        int source_idx,
        const cv::Point2f& position,
        float angle, float scale) {
        EdgeGroup transformed_group = source_group;
        cv::Point2f original_center = transformed_group.getCenter();
        cv::Point2f center_offset = cv::Point2f(-original_center.x, -original_center.y);
        transformed_group.translate(center_offset);
        if (std::abs(scale - 1.0f) > 0.01f) {
            transformed_group.scale(scale);
        }
        //if (std::abs(angle) > 0.01f) {
        //    transformed_group.rotate(angle);
        //}
        cv::Point2f final_position = position - original_center;
        transformed_group.translate(final_position);
        cv::Point2f translation = position - original_center;
        return PlacedGroup(transformed_group, source_idx, scale, angle, translation);
    }

    std::vector<PlacedGroup> TextureSynthesis::synthesizePlacement(
        const std::vector<EdgeGroup>& source_groups,
        int output_width, int output_height,
        float density,
        float angle_variation,
        float scale_variation) {

        std::vector<PlacedGroup> placed_groups;
        if (source_groups.empty() || output_width <= 0 || output_height <= 0) {
            return placed_groups;
        }

        float area_ratio = static_cast<float>(output_width * output_height) /
            (512.0f * 512.0f);
        int target_count = static_cast<int>(
            source_groups.size() * density * area_ratio * 2.0f
            );
      
        target_count = std::max(3, std::min(target_count, 50));
        std::uniform_int_distribution<int> group_dist(0,
            static_cast<int>(source_groups.size()) - 1);
        std::vector<int> usage_count(source_groups.size(), 0);

        int placed_count = 0;
        int overlap_count = 0;
        int max_attempts = 1000;
        while (placed_count < target_count && max_attempts > 0) {
            max_attempts--;
            int source_idx = group_dist(rng_);

            const EdgeGroup& source_group = source_groups[source_idx];
            cv::Point2f position = generateRandomPosition(output_width, output_height);

            float base_angle = source_group.getAverageAngle();
            float angle = generateRandomAngle(base_angle, angle_variation);
            float scale = generateRandomScale(1.0f, scale_variation);

            PlacedGroup placed_group = transformGroup(source_group, source_idx,
                position, angle, scale);
            if (avoid_overlap_ && placed_count > 0) {
                bool has_overlap = false;
                for (const auto& existing : placed_groups) {
                    if (checkOverlap(placed_group.group, existing.group, min_distance_)) {
                        has_overlap = true;
                        break;
                    }
                }
                if (has_overlap) {
                    overlap_count++;
                    continue;
                }
            }
            placed_groups.push_back(placed_group);
            usage_count[source_idx]++;
            placed_count++;
        }
        return placed_groups;
    }

    std::vector<PlacedGroup> TextureSynthesis::synthesizeFromEBPT(
        const EBPT& ebpt_model,
        int output_width, int output_height) {

        if (ebpt_model.getNumGroups() == 0) {
            return {};
        }
        const auto& source_groups = ebpt_model.getEdgeGroups();
        float density = 0.7f;
        float angle_variation = 0.3f;
        float scale_variation = 0.2f;
        return synthesizePlacement(source_groups, output_width, output_height,
            density, angle_variation, scale_variation);
    }

    cv::Mat TextureSynthesis::drawPlacementMap(
        const std::vector<PlacedGroup>& placed_groups,
        int width, int height) {

        cv::Mat placement_map = cv::Mat::zeros(height, width, CV_8UC3);
        if (placed_groups.empty()) {
            return placement_map;
        }

        std::uniform_int_distribution<int> color_dist(0, 255);
        for (size_t i = 0; i < placed_groups.size(); ++i) {
            const auto& group = placed_groups[i].group;
            cv::Scalar color = cv::Scalar(color_dist(rng_), color_dist(rng_), color_dist(rng_));
            cv::Point center(static_cast<int>(group.getCenter().x),
                static_cast<int>(group.getCenter().y));

            cv::circle(placement_map, center, 8, color, -1);
            cv::circle(placement_map, center, 3, cv::Scalar(255, 255, 255), -1);

            cv::Rect bbox = group.getBoundingBox();
            cv::rectangle(placement_map, bbox, color, 2);
            float angle = group.getAverageAngle();
            int line_length = 30;
            cv::Point direction(
                center.x + static_cast<int>(line_length * std::cos(angle)),
                center.y + static_cast<int>(line_length * std::sin(angle))
            );
            cv::line(placement_map, center, direction, color, 2);
            std::string label = std::to_string(i + 1);
            cv::putText(placement_map, label,
                cv::Point(center.x + 10, center.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 2);
            std::string edge_count = "E:" + std::to_string(group.getEdges().size());
            cv::putText(placement_map, edge_count,
                cv::Point(bbox.x, bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1);
        }
        std::string title = "Placement Map: " + std::to_string(placed_groups.size()) + " groups";
        cv::putText(placement_map, title,
            cv::Point(10, 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.7,
            cv::Scalar(255, 255, 255), 2);
        std::string size_info = "Size: " + std::to_string(width) + "x" + std::to_string(height);
        cv::putText(placement_map, size_info,
            cv::Point(10, 55),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(200, 200, 200), 1);
        return placement_map;
    }

}