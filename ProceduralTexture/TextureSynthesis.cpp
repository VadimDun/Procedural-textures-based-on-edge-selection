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
            return 0.0f;
        }
        float max_variation = variation * CV_PI;
        std::uniform_real_distribution<float> dist(-max_variation, max_variation);
        //float random_angle = base_angle + dist(rng_);
        float random_angle = dist(rng_);
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

    bool TextureSynthesis::checkHullIntersection(const std::vector<cv::Point>& hull1,
        const std::vector<cv::Point>& hull2) const {
        if (hull1.empty() || hull2.empty() || hull1.size() < 3 || hull2.size() < 3) {
            return false;
        }

        for (size_t i = 0; i < hull1.size(); ++i) {
            cv::Point p1 = hull1[i];
            cv::Point p2 = hull1[(i + 1) % hull1.size()];

            for (size_t j = 0; j < hull2.size(); ++j) {
                cv::Point q1 = hull2[j];
                cv::Point q2 = hull2[(j + 1) % hull2.size()];

                if (doSegmentsIntersect(p1, p2, q1, q2)) {
                    return true;
                }
            }
        }

        // Проверка на нахождение hull внутри другого
        if (isPointInPolygon(hull1[0], hull2) || isPointInPolygon(hull2[0], hull1)) {
            return true;
        }

        return false;
    }

    // Вспомогательная функция для проверки пересечения двух отрезков
    bool TextureSynthesis::doSegmentsIntersect(const cv::Point& p1, const cv::Point& p2,
        const cv::Point& q1, const cv::Point& q2) const {
        auto orientation = [](const cv::Point& a, const cv::Point& b, const cv::Point& c) -> int {
            int val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
            if (val == 0) return 0;  // Коллинеарны
            return (val > 0) ? 1 : 2; // 1 - по часовой, 2 - против часовой
            };

        int o1 = orientation(p1, p2, q1);
        int o2 = orientation(p1, p2, q2);
        int o3 = orientation(q1, q2, p1);
        int o4 = orientation(q1, q2, p2);

        // Общий случай
        if (o1 != o2 && o3 != o4) return true;

        // Если коллинеарны
        if (o1 == 0 && onSegment(p1, q1, p2)) return true;
        if (o2 == 0 && onSegment(p1, q2, p2)) return true;
        if (o3 == 0 && onSegment(q1, p1, q2)) return true;
        if (o4 == 0 && onSegment(q1, p2, q2)) return true;

        return false;
    }

    // Вспомогательная функция для проверки, лежит ли точка на отрезке
    bool TextureSynthesis::onSegment(const cv::Point& p, const cv::Point& q, const cv::Point& r) const {
        if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
            q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y)) {
            return true;
        }
        return false;
    }

    // Вспомогательная функция для проверки, находится ли точка внутри полигона
    bool TextureSynthesis::isPointInPolygon(const cv::Point& point, const std::vector<cv::Point>& polygon) const {
        if (polygon.size() < 3) return false;

        bool inside = false;
        for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
            const cv::Point& p1 = polygon[i];
            const cv::Point& p2 = polygon[j];

            if (((p1.y > point.y) != (p2.y > point.y)) &&
                (point.x < (p2.x - p1.x) * (point.y - p1.y) / (p2.y - p1.y) + p1.x)) {
                inside = !inside;
            }
        }

        return inside;
    }

    PlacedGroup TextureSynthesis::transformGroup(
        const SourceGroupInfo& source_info,
        const cv::Mat& input_image,
        int source_idx,
        const cv::Point2f& position,
        float angle, float scale) const {

        // Получаем bounding box группы
        cv::Rect source_bbox = source_info.group.getBoundingBox();
        cv::Point2f group_center = source_info.group.getCenter();

        // Вычисляем центр в локальных координатах bounding box
        cv::Point2f local_center(
            group_center.x - source_bbox.x,
            group_center.y - source_bbox.y
        );

        cv::Mat original_patch = input_image(source_bbox).clone();

        cv::Mat original_mask;
        if (!source_info.mask.empty()) {
            original_mask = source_info.mask(source_bbox).clone();
        }
        else {
            original_mask = cv::Mat::ones(source_bbox.size(), CV_8UC1) * 255;
        }

        cv::Mat rot_mat = cv::getRotationMatrix2D(local_center, -angle * 180.0 / CV_PI, scale);

        // Вычисляем новый размер после поворота
        cv::Rect bbox_rotated = cv::RotatedRect(local_center, original_patch.size(), -angle * 180.0 / CV_PI).boundingRect();

        // Корректируем матрицу трансформации для сохранения всего содержимого
        rot_mat.at<double>(0, 2) += (bbox_rotated.width / 2.0 - local_center.x);
        rot_mat.at<double>(1, 2) += (bbox_rotated.height / 2.0 - local_center.y);

        cv::Mat transformed_patch;
        cv::warpAffine(original_patch, transformed_patch, rot_mat,
            bbox_rotated.size(),
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        cv::Mat transformed_mask;
        cv::warpAffine(original_mask, transformed_mask, rot_mat,
            bbox_rotated.size(),
            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        // Бинаризуем маску
        cv::threshold(transformed_mask, transformed_mask, 128, 255, cv::THRESH_BINARY);

        // Вычисляем новый центр в глобальных координатах
        cv::Point2f new_center(
            position.x,
            position.y
        );

        // Вычисляем смещение для патча (верхний левый угол)
        cv::Point2f patch_offset(
            new_center.x - (bbox_rotated.width / 2.0),
            new_center.y - (bbox_rotated.height / 2.0)
        );

        // Вычисляем hull в глобальных координатах
        std::vector<cv::Point> transformed_hull;
        if (!source_info.hull.empty()) {
            transformed_hull.reserve(source_info.hull.size());

            for (const auto& point : source_info.hull) {
                cv::Point2f p(point.x, point.y);

                // Переводим в локальные координаты относительно центра группы
                p.x -= group_center.x;
                p.y -= group_center.y;

                p.x *= scale;
                p.y *= scale;

                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                float rotated_x = p.x * cos_a - p.y * sin_a;
                float rotated_y = p.x * sin_a + p.y * cos_a;

                rotated_x += new_center.x;
                rotated_y += new_center.y;

                transformed_hull.push_back(cv::Point(
                    static_cast<int>(std::round(rotated_x)),
                    static_cast<int>(std::round(rotated_y))
                ));
            }
        }

        PlacedGroup placed_group(
            transformed_patch,
            transformed_mask,
            transformed_hull,
            new_center,
            source_idx,
            scale,
            angle
        );

        return placed_group;
    }

    std::vector<PlacedGroup> TextureSynthesis::synthesizePlacement(
        const cv::Mat& input_image,
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

            PlacedGroup placed_group = transformGroup(
                source_info, input_image, source_idx, position, angle, scale);

            if (avoid_overlap_ && placed_count > 0) {
                bool has_overlap = false;

                // Проверяем пересечение с существующими группами
                for (const auto& existing : placed_groups) {
                    // Если у обеих групп есть hull, проверяем пересечение hull
                    if (!placed_group.hull.empty() && !existing.hull.empty()) {
                        if (checkHullIntersection(placed_group.hull, existing.hull)) {
                            has_overlap = true;
                            break;
                        }
                    }
                    // Если hull нет, используем расстояние между центрами
                    else {
                        float dx = placed_group.position.x - existing.position.x;
                        float dy = placed_group.position.y - existing.position.y;
                        float distance = std::sqrt(dx * dx + dy * dy);
                        float min_dist = min_distance_ * (scale + existing.scale_factor) / 2.0f;
                        if (distance < min_dist) {
                            has_overlap = true;
                            break;
                        }
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

        std::cout << "Placed groups: " << placed_groups.size()
            << " (overlaps skipped: " << overlap_count << ")" << std::endl;

        return placed_groups;
    }

}