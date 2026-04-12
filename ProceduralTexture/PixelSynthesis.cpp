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

    cv::Mat PixelSynthesis::fillPixels(
        const cv::Mat& input_image,
        const std::vector<SourceGroupInfo>& source_infos,
        const std::vector<PlacedGroup>& placed_groups,
        const cv::Size& size) {

        cv::Scalar avg_color = cv::mean(input_image);
        cv::Mat output = cv::Mat::zeros(size.height, size.width, input_image.type());
        output.setTo(avg_color);

        int groups_copied = 0;
        int groups_seamless = 0;
        int groups_simple = 0;

        for (size_t idx = 0; idx < placed_groups.size(); ++idx) {
            const auto& placed = placed_groups[idx];

            if (!placed.isValid() || placed.patch.empty()) {
                continue;
            }

            // Вычисляем bounding box патча в глобальных координатах
            int patch_left = static_cast<int>(placed.position.x - placed.patch.cols / 2);
            int patch_top = static_cast<int>(placed.position.y - placed.patch.rows / 2);
            int patch_right = patch_left + placed.patch.cols;
            int patch_bottom = patch_top + placed.patch.rows;

            // Вычисляем пересечение с выходным изображением
            int intersect_left = std::max(patch_left, 0);
            int intersect_top = std::max(patch_top, 0);
            int intersect_right = std::min(patch_right, size.width);
            int intersect_bottom = std::min(patch_bottom, size.height);

            if (intersect_left >= intersect_right || intersect_top >= intersect_bottom) {
                continue;
            }

            cv::Rect target_bbox(
                intersect_left,
                intersect_top,
                intersect_right - intersect_left,
                intersect_bottom - intersect_top
            );

            // Вычисляем ROI в патче (область, которую нужно скопировать)
            int roi_x = intersect_left - patch_left;
            int roi_y = intersect_top - patch_top;

            if (roi_x < 0 || roi_y < 0 ||
                roi_x + target_bbox.width > placed.patch.cols ||
                roi_y + target_bbox.height > placed.patch.rows) {
                std::cout << "fillPixels: ROI calculation error for group " << idx
                    << ", roi=(" << roi_x << "," << roi_y
                    << "), size=" << target_bbox.width << "x" << target_bbox.height
                    << ", patch_size=" << placed.patch.cols << "x" << placed.patch.rows
                    << std::endl;
                continue;
            }

            cv::Rect patch_roi(roi_x, roi_y, target_bbox.width, target_bbox.height);
            cv::Mat patch_part = placed.patch(patch_roi).clone();

            // Создание маски
            cv::Mat binary_mask = cv::Mat::zeros(patch_part.size(), CV_8UC1);

            if (!placed.mask.empty() &&
                placed.mask.cols == placed.patch.cols &&
                placed.mask.rows == placed.patch.rows) {
                cv::Mat mask_roi = placed.mask(patch_roi);
                cv::threshold(mask_roi, binary_mask, 128, 255, cv::THRESH_BINARY);
            }
            else if (!placed.hull.empty()) {
                std::vector<cv::Point> local_hull;
                for (const auto& p : placed.hull) {
                    cv::Point local_p(
                        p.x - target_bbox.x,
                        p.y - target_bbox.y
                    );
                    local_hull.push_back(local_p);
                }
                if (local_hull.size() >= 3) {
                    std::vector<std::vector<cv::Point>> hull_contour = { local_hull };
                    cv::fillPoly(binary_mask, hull_contour, cv::Scalar(255));
                }
                else {
                    binary_mask = cv::Mat::ones(patch_part.size(), CV_8UC1) * 255;
                }
            }
            else {
                binary_mask = cv::Mat::ones(patch_part.size(), CV_8UC1) * 255;
            }

            if (cv::countNonZero(binary_mask) == 0) {
                continue;
            }

            // Проверяем, полностью ли патч виден
            bool is_fully_visible = (patch_left >= 0 && patch_top >= 0 &&
                patch_right <= size.width && patch_bottom <= size.height);

            // Используем seamlessClone ТОЛЬКО для полностью видимых патчей
            if (is_fully_visible && patch_part.rows >= 16 && patch_part.cols >= 16) {
                try {
                    cv::Mat result;
                    cv::Mat output_roi = output(target_bbox);

                    cv::Point local_center(
                        static_cast<int>(placed.position.x - target_bbox.x),
                        static_cast<int>(placed.position.y - target_bbox.y)
                    );

                    local_center.x = std::max(0, std::min(local_center.x, target_bbox.width - 1));
                    local_center.y = std::max(0, std::min(local_center.y, target_bbox.height - 1));

                    // Дополнительная проверка центра
                    if (local_center.x >= 0 && local_center.x < binary_mask.cols &&
                        local_center.y >= 0 && local_center.y < binary_mask.rows &&
                        binary_mask.at<uchar>(local_center.y, local_center.x) != 0) {

                        cv::seamlessClone(patch_part, output_roi, binary_mask,
                            local_center, result, cv::NORMAL_CLONE);
                        result.copyTo(output(target_bbox));
                        groups_seamless++;
                    }
                    else {
                        // Центр не в маске - используем простое копирование
                        patch_part.copyTo(output(target_bbox), binary_mask);
                        groups_simple++;
                    }
                }
                catch (const cv::Exception& e) {
                    // При ошибке - простое копирование
                    patch_part.copyTo(output(target_bbox), binary_mask);
                    groups_simple++;
                }
            }
            else {
                // Для обрезанных патчей - простое копирование с размытой маской (плавный переход)
                cv::Mat soft_mask;
                cv::GaussianBlur(binary_mask, soft_mask, cv::Size(21, 21), 15.0);

                // Используем размытую маску для плавного смешивания
                for (int y = 0; y < patch_part.rows; ++y) {
                    for (int x = 0; x < patch_part.cols; ++x) {
                        if (binary_mask.at<uchar>(y, x) > 0) {
                            float alpha = soft_mask.at<uchar>(y, x) / 255.0f;
                            cv::Vec3b src = patch_part.at<cv::Vec3b>(y, x);
                            cv::Vec3b dst = output.at<cv::Vec3b>(target_bbox.y + y, target_bbox.x + x);

                            output.at<cv::Vec3b>(target_bbox.y + y, target_bbox.x + x) =
                                src * alpha + dst * (1.0f - alpha);
                        }
                    }
                }
                groups_simple++;
            }

            groups_copied++;
        }

        std::cout << "FillPixels: copied " << groups_copied
            << " groups (seamless: " << groups_seamless
            << ", blended: " << groups_simple << ")" << std::endl;

        return output;
    }

}