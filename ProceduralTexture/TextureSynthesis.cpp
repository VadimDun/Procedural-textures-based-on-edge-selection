#include "TextureSynthesis.h"
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "TextureAnalysis.h"
#include "ImageDisplay.h"

namespace EBPTns {

    TextureSynthesis::TextureSynthesis(const cv::Size& size, bool enable_rotation) {
        outputSize = size;
        std::random_device rd;
        rng_ = std::mt19937(rd());

        initScaleLevelParams(enable_rotation);
    }

    void TextureSynthesis::initScaleLevelParams(bool enable_rotation) {
        // Параметры для крупного масштаба (структура)
        ScaleLevelParams large_params;
        large_params.density = 0.4f;           // Меньше групп, но крупных
        large_params.base_scale = 1.2f;
        large_params.scale_variation = 0.3f;
        large_params.percent_fill_target = 0.4f;

        // Параметры для среднего масштаба (основная текстура)
        ScaleLevelParams medium_params;
        medium_params.density = 0.8f;
        medium_params.base_scale = 1.0f;
        medium_params.scale_variation = 0.3f;
        medium_params.percent_fill_target = 0.7f;

        // Параметры для мелкого масштаба (детали)
        ScaleLevelParams small_params;
        small_params.density = 1.2f;
        small_params.base_scale = 1.0f;
        small_params.scale_variation = 0.4f;
        small_params.percent_fill_target = 0.9f;

        if (enable_rotation) {
            large_params.angle_variation = medium_params.angle_variation = small_params.angle_variation = 0.25f;
        }
        else {
            large_params.angle_variation = medium_params.angle_variation = small_params.angle_variation = 0.0f;
        }

        scale_params_[ScaleLevel::LARGE] = large_params;
        scale_params_[ScaleLevel::MEDIUM] = medium_params;
        scale_params_[ScaleLevel::SMALL] = small_params;
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

    float TextureSynthesis::generateRandomAngle(float variation) {
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

    ////////////////////////////////

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

    ////////////////////////////////////

    void TextureSynthesis::updateOccupancyMap(const PlacedGroup& group) {
        if (occupancy_map_.empty()) return;

        // Рисуем группу на карте заполнения с весом, зависящим от уровня
        float weight = 1.0f;
        switch (group.scale_level) {
        case ScaleLevel::LARGE:  weight = 0.6f; break;  // Крупные группы дают меньший вес
        case ScaleLevel::MEDIUM: weight = 0.8f; break;
        case ScaleLevel::SMALL:  weight = 1.0f; break;  // Мелкие заполняют полностью
        }

        if (!group.hull.empty()) {
            std::vector<std::vector<cv::Point>> hull_contour = { group.hull };
            cv::fillPoly(occupancy_map_, hull_contour, cv::Scalar(weight * 255));
        }
        else {
            //cv::Rect bbox = group.group.getBoundingBox();
            //cv::rectangle(occupancy_map_, bbox, cv::Scalar(weight * 255), cv::FILLED);
            std::cerr << "TextureSynthesis::updateOccupancyMap: no hull!";
        }
    }

    float TextureSynthesis::getOccupancyAtPoint(const cv::Point2f& point) const {
        if (occupancy_map_.empty()) return 0.0f;

        int x = static_cast<int>(point.x);
        int y = static_cast<int>(point.y);

        if (x >= 0 && x < occupancy_map_.cols && y >= 0 && y < occupancy_map_.rows) {
            return occupancy_map_.at<uchar>(y, x) / 255.0f;
        }
        return 1.0f;  // За пределами изображения считаем заполненным
    }

    cv::Point2f TextureSynthesis::generatePositionByLevel(ScaleLevel level) {
        if (outputSize.width <= 0 || outputSize.height <= 0) {
            return cv::Point2f(0, 0);
        }

        float margin;
        switch (level) {
        case ScaleLevel::LARGE:  margin = 100.0f; break;
        case ScaleLevel::MEDIUM: margin = 50.0f; break;
        default: margin = 20.0f; break;
        }

        // Для всех уровней, кроме LARGE, ищем пустоты
        if (level != ScaleLevel::LARGE && !occupancy_map_.empty()) {
            std::vector<cv::Point> low_occupancy_points;

            for (int y = margin; y < outputSize.height - margin; y += margin) {
                for (int x = margin; x < outputSize.width - margin; x += margin) {
                    float occupancy = getOccupancyAtPoint(cv::Point2f(x, y));
                    float threshold = 0.5f;
                    if (occupancy < threshold) { // нет другого полигона
                        low_occupancy_points.push_back(cv::Point(x, y));
                    }
                }
            }

            if (!low_occupancy_points.empty()) {
                std::uniform_int_distribution<int> dist(0, low_occupancy_points.size() - 1);
                int idx = dist(rng_);
                return cv::Point2f(low_occupancy_points[idx].x, low_occupancy_points[idx].y);
            }
        }

        std::uniform_real_distribution<float> dist_x(margin, outputSize.width - margin);
        std::uniform_real_distribution<float> dist_y(margin, outputSize.height - margin);
        return cv::Point2f(dist_x(rng_), dist_y(rng_));
    }

    bool TextureSynthesis::checkOverlapByLevel(
        const PlacedGroup& new_group,
        const std::vector<PlacedGroup>& existing_groups,
        ScaleLevel current_level) const {

        if (existing_groups.empty()) return false;

        float new_group_area = (float)cv::contourArea(new_group.hull);
        float total_overlap_area = 0.0f;

        for (const auto& existing : existing_groups) {
            ScaleLevel existing_level = existing.scale_level;

            if (!new_group.hull.empty() && !existing.hull.empty()) {
                if (checkHullIntersection(new_group.hull, existing.hull)) {
                    float intersection_area = computeHullIntersectionArea(new_group.hull, existing.hull);
                    total_overlap_area += intersection_area;

                    // ГРУППЫ ОДНОГО УРОВНЯ - строгий запрет
                    if (current_level == existing_level) {
                        float overlap_ratio = intersection_area / new_group_area;
                        if (overlap_ratio > 0.1f) {
                            return true;
                        }
                    }
                    else if (current_level != existing_level) {
                        // Разные уровни (LARGE/MEDIUM с SMALL)
                        float overlap_ratio = intersection_area / new_group_area;
                        if (overlap_ratio > 0.25f) {
                            return true;
                        }
                    }
                }
            }
        }

        // Проверка суммарного перекрытия
        float total_overlap_ratio = total_overlap_area / new_group_area;
        if (total_overlap_ratio > 0.40f) {
            return true;
        }

        return false;
    }

    // Вспомогательная функция для вычисления площади пересечения hull
    float TextureSynthesis::computeHullIntersectionArea(
        const std::vector<cv::Point>& hull1,
        const std::vector<cv::Point>& hull2) const {

        // Переменная для хранения полигона пересечения
        std::vector<cv::Point> intersectionPolygon;

        // Вызов функции (handleNested = true для обработки случая вложенности)
        float area = cv::intersectConvexConvex(hull1, hull2, intersectionPolygon, true);

        return (area > 0) ? area : 0.0f;
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
            angle,
            source_info.scale_level
        );

        return placed_group;
    }

    std::vector<PlacedGroup> TextureSynthesis::synthesizeHierarchicalPlacement(
        const cv::Mat& input_image,
        const std::vector<SourceGroupInfo>& source_groups) {

        std::vector<PlacedGroup> all_placed_groups;

        occupancy_map_ = cv::Mat::zeros(outputSize.height, outputSize.width, CV_8UC1);

        // Группируем исходные группы по уровню
        std::map<ScaleLevel, std::vector<const SourceGroupInfo*>> groups_by_level;
        for (const auto& group_info : source_groups) {
            groups_by_level[group_info.scale_level].push_back(&group_info);
        }

        // Порядок уровней для размещения
        std::vector<ScaleLevel> order = {
            //ScaleLevel::LARGE,
            ScaleLevel::MEDIUM,
            //ScaleLevel::SMALL
        };

        for (ScaleLevel level : order) {
            if (groups_by_level[level].empty()) {
                std::cout << "\nNo " << scaleLevelToString(level) << " groups available, skipping" << std::endl;
                continue;
            }

            const auto& level_groups = groups_by_level[level];
            const auto& params = scale_params_[level];

            float current_fill = 0.0f;
            float target_fill = params.percent_fill_target;

            std::cout << "\nPlacing " << scaleLevelToString(level)
                << " groups (target fill: " << (target_fill * 100) << "%)" << std::endl;

            // Распределение для выбора исходных групп
            std::uniform_int_distribution<int> group_dist(0, level_groups.size() - 1);

            int placed_count = 0;
            int overlap_count = 0;
            int max_attempts_per_group = 1000;
            int total_attempts = 0;
            const int MAX_TOTAL_ATTEMPTS = 10000;

            while (current_fill < target_fill && total_attempts < MAX_TOTAL_ATTEMPTS) {
                total_attempts++;

                int source_idx = group_dist(rng_);
                const SourceGroupInfo& source_info = *level_groups[source_idx];

                cv::Point2f position = generatePositionByLevel(level);

                float angle = generateRandomAngle(params.angle_variation);
                float scale = generateRandomScale(params.base_scale, params.scale_variation);

                // Трансформируем группу
                PlacedGroup placed_group = transformGroup(
                    source_info, input_image, source_info.group.getIndex(), position, angle, scale);
                placed_group.scale_level = level;

                // Проверяем перекрытие с уже размещенными группами
                bool has_overlap = false;
                if (!all_placed_groups.empty()) {
                    if (checkOverlapByLevel(placed_group, all_placed_groups, level)) {
                        overlap_count++;
                        has_overlap = true;
                    }
                }

                if (has_overlap) {
                    // Если слишком много попыток подряд безуспешны, возможно, достигнут предел заполнения
                    if (overlap_count > max_attempts_per_group) {
                        std::cout << "  Too many consecutive overlaps (" << overlap_count
                            << "), moving to next level" << std::endl;
                        break;
                    }
                    continue;
                }

                // Сброс при успешном размещении
                overlap_count = 0;

                // Проверяем, что группа не слишком близко к границе
                if (position.x - placed_group.patch.cols / 2 < 0 ||
                    position.y - placed_group.patch.rows / 2 < 0 ||
                    position.x + placed_group.patch.cols / 2 > outputSize.width ||
                    position.y + placed_group.patch.rows / 2 > outputSize.height) {
                    if (placed_count % 20 == 0) {
                        std::cout << "  Group " << placed_count << " near boundary" << std::endl;
                    }
                }

                all_placed_groups.push_back(placed_group);
                updateOccupancyMap(placed_group);
                placed_count++;

                // Вычисляем текущее заполнение
                current_fill = cv::countNonZero(occupancy_map_) / (float)(occupancy_map_.total());

                if (placed_count % 10 == 0 ||
                    std::abs(current_fill - target_fill) < 0.03f) {
                    std::cout << "  Progress: placed " << placed_count
                        << " groups, current fill: " << (current_fill * 100)
                        << "%, target: " << (target_fill * 100) << "%" << std::endl;
                }

                ImageDisplay::showOccupancyMap(occupancy_map_, "Occupancy after " + scaleLevelToString(level));
            }

            std::cout << "  Finished " << scaleLevelToString(level)
                << ": placed " << placed_count << " groups"
                << ", fill: " << (current_fill * 100) << "%"
                << " (overlaps: " << overlap_count << ")" << std::endl;
        }

        float total_filled = cv::countNonZero(occupancy_map_) / (float)(occupancy_map_.total());
        std::cout << "\nFinal occupancy: " << (total_filled * 100) << "%" << std::endl;

        if (total_filled < 0.95f) {
            std::cout << "Warning: Final occupancy (" << (total_filled * 100)
                << "%) is below 95%" << std::endl;
            // todo
        }

        return all_placed_groups;
    }

}