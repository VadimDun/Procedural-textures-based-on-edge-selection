#include "TextureSynthesis.h"
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "TextureAnalysis.h"
#include "ImageDisplay.h"

namespace EBPTns {

    TextureSynthesis::TextureSynthesis(const cv::Size& size, bool enable_rotation, bool enable_scaling) {
        outputSize = size;
        std::random_device rd;
        rng_ = std::mt19937(rd());
		this->enable_rotation = enable_rotation;
        initScaleLevelParams();
    }

    void TextureSynthesis::initScaleLevelParams() {
        // Параметры для крупного масштаба (структура)
        ScaleLevelParams large_params;
        large_params.base_scale = 1.0f;
        large_params.percent_fill_target = 0.5f;

        // Параметры для среднего масштаба (основная текстура)
        ScaleLevelParams medium_params;
        medium_params.base_scale = 1.0f;
        medium_params.percent_fill_target = 0.85f;

        // Параметры для мелкого масштаба (детали)
        ScaleLevelParams small_params;
        small_params.base_scale = 1.0f;
        small_params.percent_fill_target = 0.98f;

        if (enable_scaling) {
            large_params.scale_variation = 0.2f;
            medium_params.scale_variation = 0.3f;
            small_params.scale_variation = 0.4f;
        }
        else {
            large_params.angle_variation = medium_params.angle_variation = small_params.angle_variation = 0.0f;
        }

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
        seed_ = seed;
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
            std::cerr << "TextureSynthesis::updateOccupancyMap: no hull!\n";
        }
    }

    inline void TextureSynthesis::erodeOccupancyMap(int width) {
        cv::erode(occupancy_map_, occupancy_map_, cv::Mat(), cv::Point(-1, -1), width);
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

    cv::Point2f TextureSynthesis::generatePositionForLarge() {
        if (outputSize.width <= 0 || outputSize.height <= 0) {
            return cv::Point2f(0, 0);
        }

        float margin = 100.0f;

        std::uniform_real_distribution<float> dist_x(margin, outputSize.width - margin);
        std::uniform_real_distribution<float> dist_y(margin, outputSize.height - margin);
        return cv::Point2f(dist_x(rng_), dist_y(rng_));
    }

    cv::Point TextureSynthesis::findLargestEmptyLocation(float& radius)
    {
        cv::Mat inverted;
        cv::threshold(occupancy_map_, inverted, 10, 255, cv::THRESH_BINARY);
        cv::bitwise_not(inverted,inverted);

        //ImageDisplay::show("inverted", inverted);

        cv::Mat dist;
        cv::distanceTransform(inverted, dist, cv::DIST_L2, 5);
        //ImageDisplay::show("dist", dist);

        double maxVal;
        cv::Point maxLoc;

        cv::minMaxLoc(dist, nullptr, &maxVal, nullptr, &maxLoc);

        radius = static_cast<float>(maxVal);
        return maxLoc;
    }

    PlacedGroup TextureSynthesis::transformGroup(
        const Patch& patch,
        const cv::Mat& input_image,
        uint8_t source_idx,
        const cv::Point2f& position,
        float angle, float scale) const {

        cv::Rect source_bbox = patch.bbox;

        cv::Mat original_patch = input_image(source_bbox).clone();

        cv::Point2f local_center(
            original_patch.cols / 2.0f,
            original_patch.rows / 2.0f
        );

        cv::Mat rot_mat = cv::getRotationMatrix2D(local_center, -angle * 180.0 / CV_PI, scale);

        cv::Size scaled_size(
            std::round(original_patch.cols * scale),
            std::round(original_patch.rows * scale)
        );

        cv::Rect bbox_rotated =
            cv::RotatedRect(
            local_center,
            scaled_size,
            -angle * 180.0 / CV_PI
        ).boundingRect();

        rot_mat.at<double>(0, 2) += (bbox_rotated.width / 2.0 - local_center.x);
        rot_mat.at<double>(1, 2) += (bbox_rotated.height / 2.0 - local_center.y);

        // Поворачиваем PATCH
        cv::Mat transformed_patch;
        cv::warpAffine(original_patch, transformed_patch, rot_mat,
            bbox_rotated.size(),
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        // Трансформируем hull
        std::vector<cv::Point> transformed_hull_local;

        if (!patch.hull.empty())
        {
            transformed_hull_local.reserve(patch.hull.size());

            for (const auto& point : patch.hull)
            {
                cv::Point2f p(
                    point.x - source_bbox.x,
                    point.y - source_bbox.y
                );

                std::vector<cv::Point2f> src{ p };
                std::vector<cv::Point2f> dst;

                cv::transform(src, dst, rot_mat);

                transformed_hull_local.push_back(
                    cv::Point(
                        std::round(dst[0].x),
                        std::round(dst[0].y)
                    )
                );
            }
        }

        for (const auto& p : transformed_hull_local)
        {
            if (p.x < 0 || p.y < 0)
            {
                std::cerr << "\nNEGATIVE POINT: " << p;
            }
        }

        // Строим mask из hull
        cv::Mat transformed_mask = cv::Mat::zeros(bbox_rotated.size(), CV_8UC1);

        if (transformed_hull_local.size() >= 3)
        {
            std::vector<std::vector<cv::Point>> poly{ transformed_hull_local };

            cv::fillPoly(transformed_mask, poly, cv::Scalar(255));
        }

        // Обрезаем лишнее

        if (transformed_hull_local.size() >= 3)
        {
            cv::Rect tight_bbox = cv::boundingRect(transformed_hull_local);

            // Ограничиваем bbox границами hull
            cv::Rect image_rect(
                0,
                0,
                transformed_patch.cols,
                transformed_patch.rows
            );

            tight_bbox &= image_rect;

            //std::cout << "\nTIGHT bbox = "
            //    << tight_bbox.width << "x"
            //    << tight_bbox.height << std::endl << std::endl;

            ImageDisplay::save("tr_patch_b.png", transformed_patch);
            ImageDisplay::save("tr_mask_b.png", transformed_mask);

            transformed_patch = transformed_patch(tight_bbox).clone();
            transformed_mask = transformed_mask(tight_bbox).clone();

            // Сдвигаем координаты hull
            for (auto& p : transformed_hull_local)
            {
                p.x -= tight_bbox.x;
                p.y -= tight_bbox.y;
            }
        }

        if (cv::countNonZero(transformed_mask) < MIN_SIZE_PATCH)
            return PlacedGroup();


        // Обрезаем по границе
        int patch_left = static_cast<int>(position.x - transformed_patch.cols / 2.0f);
        int patch_top = static_cast<int>(position.y - transformed_patch.rows / 2.0f);
        int patch_right = patch_left + transformed_patch.cols;
        int patch_bottom = patch_top + transformed_patch.rows;

        int clipped_left = std::max(0, patch_left);
        int clipped_top = std::max(0, patch_top);
        int clipped_right = std::min(outputSize.width, patch_right);
        int clipped_bottom = std::min(outputSize.height, patch_bottom);

        if (clipped_left >= clipped_right || clipped_top >= clipped_bottom)
        {
            return PlacedGroup();
        }

        int roi_x = clipped_left - patch_left;
        int roi_y = clipped_top - patch_top;

        int clipped_width = clipped_right - clipped_left;
        int clipped_height = clipped_bottom - clipped_top;

        cv::Rect clip_roi(
            roi_x,
            roi_y,
            clipped_width,
            clipped_height
        );

        transformed_patch = transformed_patch(clip_roi).clone();
        transformed_mask = transformed_mask(clip_roi).clone();

        for (auto& p : transformed_hull_local)
        {
            p.x -= roi_x;
            p.y -= roi_y;
        }

        // Глобальный hull

        std::vector<cv::Point> transformed_hull_global;

        transformed_hull_global.reserve(transformed_hull_local.size());

        for (const auto& p : transformed_hull_local)
        {
            int gx = p.x + clipped_left;
            int gy = p.y + clipped_top;

            transformed_hull_global.emplace_back(gx, gy);
        }

        cv::Point2f final_center(
            clipped_left + transformed_patch.cols / 2.0f,
            clipped_top + transformed_patch.rows / 2.0f
        );

        if (cv::countNonZero(transformed_mask) < MIN_SIZE_PATCH)
            return PlacedGroup();

        ImageDisplay::save("patch.png", original_patch);
        ImageDisplay::save("tr_patch_a.png", transformed_patch);
        ImageDisplay::save("tr_mask_a.png", transformed_mask);

        PlacedGroup placed_group(
            transformed_patch,
            transformed_mask,
            transformed_hull_global,
            final_center,
            source_idx,
            angle,
            patch.scale_level
        );

        return placed_group;
    }

    std::vector<PlacedGroup> TextureSynthesis::synthesizeHierarchicalPlacement(
        const cv::Mat& input_image,
        const std::vector<Patch>& patches) {

        std::vector<PlacedGroup> all_placed_groups;
        auto total_start = std::chrono::high_resolution_clock::now();

        occupancy_map_ = cv::Mat::zeros(outputSize.height, outputSize.width, CV_8UC1);

        // Группируем исходные группы по уровню
        std::map<ScaleLevel, std::vector<const Patch*>> groups_by_level;
        for (const auto& patch : patches) {
            groups_by_level[patch.scale_level].push_back(&patch);
        }

        // Порядок уровней для размещения
        std::vector<ScaleLevel> order = {
            ScaleLevel::LARGE,
            ScaleLevel::MEDIUM,
            ScaleLevel::SMALL
        };

        for (ScaleLevel level : order) {
            if (groups_by_level[level].empty()) {
                std::cout << "\nNo " << scaleLevelToString(level) << " groups available, skipping" << std::endl;
                continue;
            }

            auto level_start = std::chrono::high_resolution_clock::now();
            if (level == ScaleLevel::MEDIUM) {
                erodeOccupancyMap(20);                               
            }
            else if (level == ScaleLevel::SMALL) {
                erodeOccupancyMap(3);
            }

            const auto& level_groups = groups_by_level[level];
            const auto& params = scale_params_[level];

            float current_fill = 0.0f;
            float target_fill = params.percent_fill_target;

            std::cout << "\nPlacing " << scaleLevelToString(level)
                << " groups (target fill: " << (target_fill * 100) << "%)" << std::endl;

            // Распределение для выбора исходных групп
            std::uniform_int_distribution<short> group_dist(0, level_groups.size() - 1);

            int placed_count = 0;
            int overlap_count = 0;
            int max_attempts_per_group = 1000;
            int total_attempts = 0;
            const int MAX_TOTAL_ATTEMPTS = 1000;

            while (current_fill < target_fill && total_attempts < MAX_TOTAL_ATTEMPTS) {
                total_attempts++;

                const Patch* patch;

                cv::Point2f position;
                float scale;

                if (level != ScaleLevel::LARGE)
                {
                    float radius;
                    position = findLargestEmptyLocation(radius);
                    //float min_group_size = level_groups[level_groups.size() - 1]->radial_spread_;
                    //if (radius < min_group_size * 0.3)
                    //    break;

                    float desired_size = radius * 2.0f;

                    uint8_t best_idx = 0;
                    float best_diff = 1e9f;

                    for (int i = 0; i < level_groups.size(); i++)
                    {
                        float group_size = level_groups[i]->radial_spread_;

                        float diff = std::abs(group_size - desired_size);

                        if (diff < best_diff)
                        {
                            best_diff = diff;
                            best_idx = i;
                        }
                    }

                    if (1.3 * level_groups[0]->radial_spread_ < radius) {
                        best_idx = group_dist(rng_);
                    }

                    patch = level_groups[best_idx];

                    float base_patch_size = static_cast<float>(patch->radial_spread_);

                    scale = desired_size / base_patch_size;
                    scale = std::max(0.3f, std::min(scale, 1.2f));
                }
                else
                {
                    uint8_t source_idx = group_dist(rng_);
                    patch = level_groups[source_idx];

                    position = generatePositionForLarge();
                    scale = generateRandomScale(params.base_scale, params.scale_variation);
                }

                float angle = generateRandomAngle(params.angle_variation);

                // Трансформируем группу
                PlacedGroup placed_group = transformGroup(
                    *patch, input_image, patch->index, position, angle, scale);
                if (placed_group.source_index == -1) continue;

                placed_group.scale_level = level;

                all_placed_groups.push_back(placed_group);
                //return all_placed_groups;

                updateOccupancyMap(placed_group);
                placed_count++;

                // Вычисляем текущее заполнение
                current_fill = cv::countNonZero(occupancy_map_) / (float)(occupancy_map_.total());

                //if (placed_count % 10 == 0 ||
                //    std::abs(current_fill - target_fill) < 0.03f) {
                //    std::cout << "  Progress: placed " << placed_count
                //        << " groups, current fill: " << (current_fill * 100)
                //        << "%, target: " << (target_fill * 100) << "%" << std::endl;
                //}

                ImageDisplay::showOccupancyMap(occupancy_map_, "Occupancy after " + scaleLevelToString(level));
            }

            auto level_end = std::chrono::high_resolution_clock::now();
            auto level_duration = std::chrono::duration_cast<std::chrono::milliseconds>(level_end - level_start);

            std::cout << "  Finished " << scaleLevelToString(level)
                << ": placed " << placed_count << " groups"
                << ", fill: " << (current_fill * 100) << "%"
                << ", time: " << level_duration.count() / 1000.0 << " sec" << std::endl;
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

        float total_filled = cv::countNonZero(occupancy_map_) / (float)(occupancy_map_.total());
        std::cout << "\nFinal occupancy: " << (total_filled * 100) << "%" << std::endl;
        std::cout << "Total synthesis time: " << total_duration.count() / 1000.0 << " sec" << std::endl;

        if (total_filled < 0.95f) {
            std::cout << "Warning: Final occupancy (" << (total_filled * 100)
                << "%) is below 95%" << std::endl;
            // todo
        }

        return all_placed_groups;
    }

}