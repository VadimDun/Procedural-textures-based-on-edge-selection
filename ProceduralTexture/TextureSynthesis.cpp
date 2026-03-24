#include "TextureSynthesis.h"
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "TextureAnalysis.h"

namespace EBPTns {

    TextureSynthesis::TextureSynthesis(const cv::Size& size) {
        outputSize = size;
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }

    void TextureSynthesis::setRandomSeed(unsigned int seed) {
        rng_.seed(seed);
    }

    cv::Point2f TextureSynthesis::generateRandomPosition() {
        if (outputSize.width <= 0 || outputSize.height <= 0) {
            return cv::Point2f(0, 0);
        }
        float margin = 50.0f;
        std::uniform_real_distribution<float> dist_x(
            margin,
            std::max(margin, static_cast<float>(outputSize.width) - margin)
        );
        std::uniform_real_distribution<float> dist_y(
            margin,
            std::max(margin, static_cast<float>(outputSize.height) - margin)
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

    PlacedGroup TextureSynthesis::transformGroup(const SourceGroupInfo& source_info,
        int source_idx,
        const cv::Point2f& position,
        float angle, float scale) const {

        EdgeGroup transformed_group = source_info.group;

        // Сохраняем оригинальный центр
        cv::Point2f original_center = transformed_group.getCenter();

        // Центрируем группу
        cv::Point2f center_offset = cv::Point2f(-original_center.x, -original_center.y);
        transformed_group.translate(center_offset);

        // Применяем трансформации
        if (std::abs(scale - 1.0f) > 0.01f) {
            transformed_group.scale(scale);
        }

        if (std::abs(angle) > 0.01f) {
            transformed_group.rotate(angle);
        }

        // Перемещаем в целевую позицию
        transformed_group.translate(position);

        // Вычисляем полное смещение
        cv::Point2f translation = position - original_center;

        PlacedGroup placed_group(transformed_group, source_idx,
            scale, angle, translation);

        placed_group.mask = TextureAnalysis::getMask(transformed_group, outputSize, placed_group.hull);

        return placed_group;
    }

    std::vector<PlacedGroup> TextureSynthesis::synthesizePlacement(
        const std::vector<SourceGroupInfo>& source_groups,
        float density,
        float angle_variation,
        float scale_variation) {

        std::vector<PlacedGroup> placed_groups;

        if (source_groups.empty() || outputSize.width <= 0 || outputSize.height <= 0) {
            std::cerr << "Error: empty source groups or invalid output size" << std::endl;
            return placed_groups;
        }

        std::cout << "Synthesizing placement with " << source_groups.size() << " source groups" << std::endl;

        // Рассчитываем количество групп для размещения
        float area_ratio = static_cast<float>(outputSize.width * outputSize.height) / (512.0f * 512.0f);
        int target_count = static_cast<int>(source_groups.size() * density * area_ratio * 2.0f);
        //int target_count = static_cast<int>(source_groups.size() * density * 2.0f);
        target_count = std::max(3, std::min(target_count, 50));

        std::cout << "Target groups: " << target_count << std::endl;

        // Распределение для выбора исходных групп
        std::uniform_int_distribution<int> group_dist(0, static_cast<int>(source_groups.size()) - 1);

        int placed_count = 0;
        int overlap_count = 0;
        int max_attempts = 1000;

        while (placed_count < target_count && max_attempts > 0) {
            max_attempts--;

            int source_idx = group_dist(rng_);
            const SourceGroupInfo& source_info = source_groups[source_idx];

            cv::Point2f position = generateRandomPosition();
            float angle = generateRandomAngle(source_info.group.getAverageAngle(), angle_variation);
            float scale = generateRandomScale(1.0f, scale_variation);

            PlacedGroup placed_group = transformGroup(source_info, source_idx, position, angle, scale);

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
            placed_count++;
        }

        std::cout << "Placed groups: " << placed_groups.size() << " (overlaps skipped: " << overlap_count << ")" << std::endl;

        return placed_groups;
    }

}