#include "PixelSynthesis.h"
#include "ImageDisplay.h"
#include <random>
#include <iostream>

namespace EBPTns {

    void PixelSynthesis::GaussianCopy(const cv::Mat& binary_mask, const cv::Mat& patch_part, cv::Mat& output, const cv::Rect& target_bbox) {
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
            int i = idx + 1;

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

            int clipped_width = intersect_right - intersect_left;
            int clipped_height = intersect_bottom - intersect_top;

            //std::cout << "patch size: << placed.patch.cols
            //    << " expected width: << clipped_width << std::endl;

            if (clipped_width <= 0 || clipped_height <= 0)
                continue;

            // ROI внутри patch
            int roi_x = intersect_left - patch_left;
            int roi_y = intersect_top - patch_top;

            cv::Rect patch_roi(
                roi_x,
                roi_y,
                clipped_width,
                clipped_height
            );

            cv::Mat patch_part = placed.patch(patch_roi).clone();

            cv::Rect target_bbox(
                intersect_left,
                intersect_top,
                clipped_width,
                clipped_height
            );

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

                    // координаты hull внутри исходного patch
                    int patch_local_x = p.x - patch_left;
                    int patch_local_y = p.y - patch_top;

                    // учитываем clipping
                    int clipped_x = patch_local_x - roi_x;
                    int clipped_y = patch_local_y - roi_y;

                    // проверяем, что точка внутри patch_part
                    if (clipped_x >= 0 && clipped_x < patch_part.cols &&
                        clipped_y >= 0 && clipped_y < patch_part.rows)
                    {
                        local_hull.emplace_back(clipped_x, clipped_y);
                    }
                }

                if (local_hull.size() >= 3) {
                    std::vector<std::vector<cv::Point>> hull_contour = { local_hull };
                    cv::fillPoly(binary_mask, hull_contour, cv::Scalar(255));
                }
                else {
                    binary_mask.setTo(255);
                }
            }
            else {
                binary_mask.setTo(255);
            }

            if (cv::countNonZero(binary_mask) == 0) {
                continue;
            }
            bool isSmall = (patch_part.rows < 40 || patch_part.cols < 40);
            //bool isSmall = placed.scale_level == ScaleLevel::SMALL;
            // С маленькими ошибка вылетает
            if (!isSmall) {
            //if (true) {
                try {
                    cv::Mat result;
                    cv::Mat output_roi = output(target_bbox);
                    //if (i == 3) ImageDisplay::show("output_roi", output_roi);
                    cv::Point local_center(
                        static_cast<int>(clipped_width / 2),
                        static_cast<int>(clipped_height / 2)
                    );

                    local_center.x = std::max(0, std::min(local_center.x, target_bbox.width - 1));
                    local_center.y = std::max(0, std::min(local_center.y, target_bbox.height - 1));

                    // Дополнительная проверка центра
                    if (local_center.x >= 0 && local_center.x < binary_mask.cols &&
                        local_center.y >= 0 && local_center.y < binary_mask.rows &&
                        binary_mask.at<uchar>(local_center.y, local_center.x) != 0) {
                        //std::cout << scaleLevelToString(placed.scale_level);
                        auto clone_mode = placed.scale_level == ScaleLevel::SMALL ? cv::MIXED_CLONE : cv::NORMAL_CLONE;
                        cv::seamlessClone(patch_part, output_roi, binary_mask, local_center, result, clone_mode);
                        result.copyTo(output(target_bbox));
                        //if (i == 3) { ImageDisplay::show("output", output); ImageDisplay::show("result", result);}

                        groups_seamless++;
                    }
                    else {
                        // Центр не в маске - используем простое копирование
                        GaussianCopy(binary_mask, patch_part, output, target_bbox);
                        groups_simple++;
                    }
                }
                catch (const cv::Exception& e) {
                    // При ошибке - копирование
                    GaussianCopy(binary_mask, patch_part, output, target_bbox);
                    groups_simple++;
                    if (i == 3) 
                    {
                        // Вывод отладочной информации
                        std::cout << "CV Exception caught for group " << i << ":" << std::endl;
                        std::cout << "  Exception: " << e.what() << std::endl;
                        std::cout << "  target_bbox: " << target_bbox << std::endl;
                        std::cout << "  binary_mask size: " << binary_mask.cols
                            << "x" << binary_mask.rows << std::endl;
                        std::cout << "  patch_part size: " << patch_part.cols
                            << "x" << patch_part.rows << std::endl;
                        std::cout << " intersect_left=" << intersect_left
                            << " intersect_right=" << intersect_right
                            << "\n intersect_top=" << intersect_top
                            << " intersect_bottom=" << intersect_bottom
                            << "\n clipped_width=" << clipped_width
                            << " clipped_height=" << clipped_height
                            << "\n roi_x=" << roi_x
                            << " roi_y=" << roi_y << std::endl << std::endl;
                        std::cout << "  patch_left: " << patch_left << ", patch_top: " << patch_top << std::endl;
                        std::cout << "  patch_right: " << patch_right << ", patch_bottom: " << patch_bottom << std::endl;
                    }
                }
            }
            else {
                // Для обрезанных патчей - простое копирование с размытой маской (плавный переход)
                GaussianCopy(binary_mask, patch_part, output, target_bbox);
                groups_simple++;
                std::cout << i <<" patch_part.rows=" << patch_part.rows << " patch_part.cols=" << patch_part.cols << std::endl;
            }

            //if (i == 3)
            //{
            //    std::cout << "Debug group " << i << ":" << std::endl;
            //    std::cout << "  position: " << placed.position << std::endl;
            //    std::cout << "  patch size: " << placed.patch.size() << std::endl;
            //    std::cout << "  hull size: " << placed.hull.size() << std::endl;
            //    std::cout << "  patch_left: " << patch_left << ", patch_top: " << patch_top << std::endl;
            //    std::cout << "  patch_right: " << patch_right << ", patch_bottom: " << patch_bottom << std::endl;
            //    std::cout << "  target_bbox: " << target_bbox << std::endl;
            //    std::cout << "  roi: " << roi_x << ", " << roi_y << std::endl
            //        << "=======================================" << std::endl << std::endl;

            //    // Сохраняем для визуального контроля
            //    std::string s = "Patch_part" + std::to_string(i);
            //    std::string s1 = "Mask_part" + std::to_string(i);
            //    std::string ss = "Patch" + std::to_string(i);
            //    std::string ss1 = "Mask" + std::to_string(i);
            //    //ImageDisplay::show(s, patch_part);
            //    ImageDisplay::show(s, patch_part);
            //    ImageDisplay::show(s1, binary_mask);
            //    ImageDisplay::show(ss, placed.patch);
            //    ImageDisplay::show(ss1, placed.mask);
            //}

            groups_copied++;
        }

        std::cout << "FillPixels: copied " << groups_copied
            << " groups (seamless: " << groups_seamless
            << ", blended: " << groups_simple << ")" << std::endl;

        return output;
    }

}