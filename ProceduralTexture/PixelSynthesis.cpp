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

        for (size_t idx = 0; idx < placed_groups.size(); ++idx) {
            const auto& placed = placed_groups[idx];

            if (!placed.isValid() || placed.patch.empty()) {
                continue;
            }

            // Вычисляем позицию левого верхнего угла патча
            int patch_left = static_cast<int>(placed.position.x - placed.patch.cols / 2);
            int patch_top = static_cast<int>(placed.position.y - placed.patch.rows / 2);

            // Вычисляем пересечение с выходным изображением
            cv::Rect target_bbox(
                std::max(0, patch_left),
                std::max(0, patch_top),
                placed.patch.cols,
                placed.patch.rows
            );

            // Обрезаем по границам
            if (target_bbox.x + target_bbox.width > size.width) {
                target_bbox.width = size.width - target_bbox.x;
            }
            if (target_bbox.y + target_bbox.height > size.height) {
                target_bbox.height = size.height - target_bbox.y;
            }

            if (target_bbox.width <= 0 || target_bbox.height <= 0) continue;

            // Вычисляем ROI в патче
            int roi_x = target_bbox.x - patch_left;
            int roi_y = target_bbox.y - patch_top;

            if (roi_x < 0 || roi_y < 0 ||
                roi_x + target_bbox.width > placed.patch.cols ||
                roi_y + target_bbox.height > placed.patch.rows) {
                continue;
            }

            cv::Rect patch_roi(roi_x, roi_y, target_bbox.width, target_bbox.height);
            cv::Mat patch_part = placed.patch(patch_roi);

            // Получаем маску
            cv::Mat mask_part;
            if (!placed.mask.empty() &&
                placed.mask.cols == placed.patch.cols &&
                placed.mask.rows == placed.patch.rows) {
                mask_part = placed.mask(patch_roi);
            }
            else {
                // Создаем маску из hull
                mask_part = cv::Mat::zeros(patch_part.size(), CV_8UC1);
                if (!placed.hull.empty()) {
                    std::vector<cv::Point> local_hull;
                    for (const auto& p : placed.hull) {
                        cv::Point local_p(
                            p.x - target_bbox.x,
                            p.y - target_bbox.y
                        );
                        if (local_p.x >= 0 && local_p.x < mask_part.cols &&
                            local_p.y >= 0 && local_p.y < mask_part.rows) {
                            local_hull.push_back(local_p);
                        }
                    }
                    if (local_hull.size() >= 3) {
                        std::vector<std::vector<cv::Point>> hull_contour = { local_hull };
                        cv::fillPoly(mask_part, hull_contour, cv::Scalar(255));
                    }
                    else {
                        mask_part = cv::Mat::ones(patch_part.size(), CV_8UC1) * 255;
                    }
                }
                else {
                    mask_part = cv::Mat::ones(patch_part.size(), CV_8UC1) * 255;
                }
            }

            cv::Mat binary_mask;
            cv::threshold(mask_part, binary_mask, 128, 255, cv::THRESH_BINARY);

            if (cv::countNonZero(binary_mask) == 0) continue;

            // Вычисляем локальный центр (центр группы относительно target_bbox)
            cv::Point local_center(
                static_cast<int>(placed.position.x - target_bbox.x),
                static_cast<int>(placed.position.y - target_bbox.y)
            );

            local_center.x = std::max(0, std::min(local_center.x, target_bbox.width - 1));
            local_center.y = std::max(0, std::min(local_center.y, target_bbox.height - 1));

            // Вставляем патч
            bool use_seamless = (patch_part.rows >= 6 && patch_part.cols >= 6 &&
                binary_mask.rows >= 6 && binary_mask.cols >= 6);

            if (use_seamless) {
                try {
                    cv::Mat result;
                    cv::Mat output_roi = output(target_bbox);
                    cv::seamlessClone(patch_part, output_roi, binary_mask,
                        local_center, result, cv::NORMAL_CLONE);
                    result.copyTo(output(target_bbox));
                }
                catch (const cv::Exception& e) {
                    patch_part.copyTo(output(target_bbox), binary_mask);
                }
            }
            else 
            {
                patch_part.copyTo(output(target_bbox), binary_mask);
            }
        }

        return output;
    }

}