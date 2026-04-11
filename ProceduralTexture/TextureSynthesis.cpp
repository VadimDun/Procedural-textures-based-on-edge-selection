#include "TextureSynthesis.h"
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
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

        // Параметры для среднего масштаба (основная текстура)
        ScaleLevelParams medium_params;
        medium_params.density = 0.8f;
        medium_params.base_scale = 1.0f;
        medium_params.scale_variation = 0.3f;

        // Параметры для мелкого масштаба (детали)
        ScaleLevelParams small_params;
        small_params.density = 1.2f;
        small_params.base_scale = 0.85f;
        small_params.scale_variation = 0.4f;

        // Сверхмелкий масштаб
        ScaleLevelParams fine_params;
        fine_params.density = 1.5f;
        fine_params.base_scale = 1.0f;
        fine_params.scale_variation = 0.5f;

        if (enable_rotation) {
            large_params.angle_variation = medium_params.angle_variation = small_params.angle_variation = 0.25f;
            fine_params.angle_variation = 0.5f;
        }
        else {
            large_params.angle_variation = medium_params.angle_variation = small_params.angle_variation = fine_params.angle_variation = 0.0f;
        }

        scale_params_[ScaleLevel::LARGE] = large_params;
        scale_params_[ScaleLevel::MEDIUM] = medium_params;
        scale_params_[ScaleLevel::SMALL] = small_params;
        scale_params_[ScaleLevel::FINE] = fine_params;

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

    void TextureSynthesis::classifySourceGroups(std::vector<SourceGroupInfo>& source_groups) {
        const bool USE_RADIAL_SPRED = true;
        if (source_groups.empty()) return;

        std::vector<float> group_sizes;
        group_sizes.reserve(source_groups.size());

        for (const auto& group_info : source_groups) {
            float size = USE_RADIAL_SPRED ? group_info.group.getRadialSpread() : cv::contourArea(group_info.hull);
            if (size == 0) continue;
            group_sizes.push_back(size);
        }

        std::vector<float> sorted_sizes = group_sizes;
        std::sort(sorted_sizes.begin(), sorted_sizes.end());

        size_t n = sorted_sizes.size();

        // СТРАТЕГИЯ 1: Поиск естественных разрывов
        std::vector<float> gaps;

        // Находим наибольшие разрывы между последовательными размерами
        for (size_t i = 1; i < n; ++i) {
            float gap = sorted_sizes[i] - sorted_sizes[i - 1];
            gaps.push_back(gap);
        }

        // Сортируем разрывы по убыванию
        std::vector<int> sorted_indices(gaps.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&gaps](int a, int b) { return gaps[a] > gaps[b]; });

        //for (size_t i = 0; i < n-1; ++i) {
        //    std::cout << "sorted gap " << i << ": " << gaps[sorted_indices[i]] << std::endl;
        //}

        // Используем 2 наибольших разрыва для разделения на 3 категории (крупные/средние/мелкие)
        if (n >= 4 && sorted_indices.size() >= 2) {
            int gap1_idx = sorted_indices[0];
            int split1 = gap1_idx + 1;

            int gap2_idx = sorted_indices[1];
            int split2 = gap2_idx + 1;

            // взяли два самых больших разрыва. разрыв с меньшим индексом разделяет large и 
            // medium(так как меньший индекс ближе к началу в отсортированном по размеру массиве)
            int lower_split = std::min(split1, split2);
            int upper_split = std::max(split1, split2);

            // Устанавливаем пороги на основе разрывов
            large_scale_threshold_ = sorted_sizes[upper_split];
            medium_scale_threshold_ = sorted_sizes[lower_split];

            // Для FINE используем самый маленький размер или порог
            small_scale_threshold_ = sorted_sizes[0];

            std::cout << "Using natural breakpoints strategy:" << std::endl;
        }
        // СТРАТЕГИЯ 2: На основе среднего и стандартного отклонения
        else if (n >= 3) {
            float mean = 0.0f;
            for (float s : group_sizes) mean += s;
            mean /= n;

            float variance = 0.0f;
            for (float s : group_sizes) {
                variance += (s - mean) * (s - mean);
            }
            variance /= n;
            float std_dev = std::sqrt(variance);

            // Пороги: mean ± 0.5*std_dev (можно настроить коэффициент)
            large_scale_threshold_ = mean + 0.5f * std_dev;
            medium_scale_threshold_ = mean - 0.5f * std_dev;
            small_scale_threshold_ = mean - 1.0f * std_dev;

            std::cout << "Using statistical thresholds (mean=" << mean
                << ", std=" << std_dev << "):" << std::endl;
        }
        else {
            float min_size = sorted_sizes[0];
            float max_size = sorted_sizes[n - 1];
            float range = max_size - min_size;

            large_scale_threshold_ = min_size + range * 0.75f;
            medium_scale_threshold_ = min_size + range * 0.5f;
            small_scale_threshold_ = min_size + range * 0.25f;
            std::cout << "Using area:" << std::endl;
        }

        std::cout << "Scale thresholds: LARGE >= " << large_scale_threshold_
            << ", MEDIUM >= " << medium_scale_threshold_
            << ", SMALL >= " << small_scale_threshold_
            << ", FINE < " << small_scale_threshold_ << std::endl;

        // Классифицируем каждую группу
        for (auto& group_info : source_groups) {
            float size = USE_RADIAL_SPRED ? group_info.group.getRadialSpread() : cv::contourArea(group_info.hull);

            if (size >= large_scale_threshold_) {
                group_info.scale_level = ScaleLevel::LARGE;
            }
            else if (size >= medium_scale_threshold_) {
                group_info.scale_level = ScaleLevel::MEDIUM;
            }
            else if (size >= small_scale_threshold_) {
                group_info.scale_level = ScaleLevel::SMALL;
            }
            else {
                group_info.scale_level = ScaleLevel::FINE;
            }

            std::cout << "Group " << (&group_info - &source_groups[0])
                << ": size=" << size
                << ", level=" << scaleLevelToString(group_info.scale_level) << std::endl;
        }

        checkAndAdjustThresholds(source_groups);
    }

    // TODO сделать проверку каждой группы
    // Вспомогательный метод для проверки и коррекции
    void TextureSynthesis::checkAndAdjustThresholds(std::vector<SourceGroupInfo>& source_groups) {
        int large_count = 0, medium_count = 0, small_count = 0, fine_count = 0;

        for (const auto& group : source_groups) {
            switch (group.scale_level) {
            case ScaleLevel::LARGE: large_count++; break;
            case ScaleLevel::MEDIUM: medium_count++; break;
            case ScaleLevel::SMALL: small_count++; break;
            case ScaleLevel::FINE: fine_count++; break;
            }
        }

        // Если более 80% групп в одном уровне, корректируем пороги
        int total = source_groups.size();
        if (large_count > total * 0.8f) {
            std::cout << "Warning: Too many LARGE groups, adjusting thresholds..." << std::endl;
            large_scale_threshold_ *= 0.7f;
            medium_scale_threshold_ *= 0.7f;
            // Переклассифицируем
            for (auto& group : source_groups) {
                float size = group.group.getRadialSpread();
                if (size >= large_scale_threshold_) {
                    group.scale_level = ScaleLevel::LARGE;
                }
                else if (size >= medium_scale_threshold_) {
                    group.scale_level = ScaleLevel::MEDIUM;
                }
                else {
                    group.scale_level = ScaleLevel::SMALL;
                }
            }
        }
    }

    void TextureSynthesis::updateOccupancyMap(const PlacedGroup& group) {
        if (occupancy_map_.empty()) return;

        // Рисуем группу на карте заполнения с весом, зависящим от уровня
        float weight = 1.0f;
        switch (group.scale_level) {
        case ScaleLevel::LARGE:  weight = 0.6f; break;  // Крупные группы дают меньший вес
        case ScaleLevel::MEDIUM: weight = 0.8f; break;
        case ScaleLevel::SMALL:  weight = 1.0f; break;  // Мелкие заполняют полностью
        case ScaleLevel::FINE:   weight = 1.2f; break;
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
        case ScaleLevel::LARGE:  margin = 100.0f; break;  // Крупные группы дальше от края
        case ScaleLevel::MEDIUM: margin = 50.0f; break;
        case ScaleLevel::SMALL:  margin = 30.0f; break;
        default:   margin = 20.0f; break;
        }

        // Для уровней MEDIUM и SMALL - предпочитаем менее заполненные области
        if (level != ScaleLevel::LARGE) {
            // Используем выборку по важности на основе occupancy_map
            if (!occupancy_map_.empty()) {
                // Находим области с низкой заполненностью
                std::vector<cv::Point> low_occupancy_points;

                for (int y = margin; y < outputSize.height - margin; y += margin) {
                    for (int x = margin; x < outputSize.width - margin; x += margin) {
                        float occupancy = getOccupancyAtPoint(cv::Point2f(x, y));
                        if (occupancy < 0.5f) {  // нет другого полигона
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

        const auto& params = scale_params_.at(current_level);

        for (const auto& existing : existing_groups) {
            ScaleLevel existing_level = existing.scale_level;

            if (!new_group.hull.empty() && !existing.hull.empty()) {
                if (checkHullIntersection(new_group.hull, existing.hull)) {
                    // Проверяем степень пересечения (площадь пересечения)
                    float intersection_area = computeHullIntersectionArea(new_group.hull, existing.hull);
                    //std::cout << "\n inter area = " << intersection_area << std::endl;
                    float new_group_area = (float)cv::contourArea(new_group.hull);
                    float overlap_ratio = intersection_area / new_group_area;

                    if (current_level != existing_level) {
                        if (overlap_ratio < 0.3f) {
                            continue;
                        }
                    }
                    else {
                        if (overlap_ratio < 0.3f) {
                            continue;
                        }
                    }
                    return true;  // Пересечение слишком большое
                }
            }
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

    void TextureSynthesis::setScaleThresholds(float large, float medium, float small) {
        large_scale_threshold_ = large;
        medium_scale_threshold_ = medium;
        small_scale_threshold_ = small;
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
            float angle = generateRandomAngle(angle_variation);
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

    std::vector<PlacedGroup> TextureSynthesis::synthesizeHierarchicalPlacement(
        const cv::Mat& input_image,
        const std::vector<SourceGroupInfo>& source_groups) {

        std::vector<PlacedGroup> all_placed_groups;

        occupancy_map_ = cv::Mat::zeros(outputSize.height, outputSize.width, CV_8UC1);

        std::vector<SourceGroupInfo> classified_groups = source_groups;
        classifySourceGroups(classified_groups); // todo вынести в анализ

        // Группируем исходные группы по уровню
        std::map<ScaleLevel, std::vector<const SourceGroupInfo*>> groups_by_level;
        for (const auto& group_info : classified_groups) {
            groups_by_level[group_info.scale_level].push_back(&group_info);
        }

        // Порядок уровней для размещения
        std::vector<ScaleLevel> order = {
            ScaleLevel::LARGE,
            ScaleLevel::MEDIUM,
            ScaleLevel::SMALL,
            ScaleLevel::FINE
        };

        for (ScaleLevel level : order) {
            if (groups_by_level[level].empty()) continue;

            const auto& level_groups = groups_by_level[level];
            const auto& params = scale_params_[level];

            // Рассчитываем целевое количество групп для этого уровня
            float area_ratio = static_cast<float>(outputSize.width * outputSize.height) / (512.0f * 512.0f);
            int target_count = static_cast<int>(level_groups.size() * params.density * area_ratio);
            target_count = std::max(3, std::min(target_count, 30));

            std::cout << "\nPlacing " << scaleLevelToString(level)
                << " groups: target=" << target_count
                << ", available=" << level_groups.size() << std::endl;

            // Распределение для выбора исходных групп
            std::uniform_int_distribution<int> group_dist(0, level_groups.size() - 1);

            int placed_count = 0;
            int overlap_count = 0;
            int max_attempts = 500;

            while (placed_count < target_count && max_attempts > 0) {
                max_attempts--;

                int source_idx = group_dist(rng_);
                const SourceGroupInfo& source_info = *level_groups[source_idx];

                cv::Point2f position = generatePositionByLevel(level);

                float angle = generateRandomAngle(params.angle_variation);
                float scale = generateRandomScale(params.base_scale, params.scale_variation);

                // Трансформируем группу
                PlacedGroup placed_group = transformGroup(
                    source_info, input_image, source_idx, position, angle, scale);
                placed_group.scale_level = level;

                // Проверяем перекрытие с уже размещенными группами
                if (!all_placed_groups.empty()) {
                    if (checkOverlapByLevel(placed_group, all_placed_groups, level)) {
                        overlap_count++;
                        continue;
                    }
                }

                // Проверяем, что группа не слишком близко к границе
                if (position.x - placed_group.patch.cols / 2 < 0 ||
                    position.y - placed_group.patch.rows / 2 < 0 ||
                    position.x + placed_group.patch.cols / 2 > outputSize.width ||
                    position.y + placed_group.patch.rows / 2 > outputSize.height) {
                    // Группа на границе - все равно размещаем, но с предупреждением
                    if (overlap_count % 50 == 0) {
                        std::cout << "  Group " << placed_count << " near boundary" << std::endl;
                    }
                }

                all_placed_groups.push_back(placed_group);
                updateOccupancyMap(placed_group);
                placed_count++;

                //if (placed_count % 10 == 0) {
                //    std::cout << "  Progress: " << placed_count << "/" << target_count << std::endl;
                //}

                ImageDisplay::showOccupancyMap(occupancy_map_, "Occupancy after " + scaleLevelToString(level));
            }

            std::cout << "  Placed " << placed_count << " " << scaleLevelToString(level)
                << " groups (overlaps: " << overlap_count << ")" << std::endl;
        }

        float total_filled = cv::countNonZero(occupancy_map_) / (float)(occupancy_map_.total());
        std::cout << "\nFinal occupancy: " << (total_filled * 100) << "%" << std::endl;

        if (total_filled < 0.95f) {
            std::cout << "Attempting to fill remaining gaps..." << std::endl;
            // todo
        }

        return all_placed_groups;
    }

}