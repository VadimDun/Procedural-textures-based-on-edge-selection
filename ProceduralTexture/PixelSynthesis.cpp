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
        const std::vector<PlacedGroup>& placed_groups,
        const cv::Size& size) {

        auto total_start = std::chrono::high_resolution_clock::now();

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

            int patch_left = static_cast<int>(placed.position.x - placed.patch.cols / 2);

            int patch_top = static_cast<int>(placed.position.y - placed.patch.rows / 2);

            cv::Rect target_bbox(
                patch_left,
                patch_top,
                placed.patch.cols,
                placed.patch.rows
            );

            cv::Mat patch_part = placed.patch;

            // Создание маски
            cv::Mat binary_mask = cv::Mat::zeros(patch_part.size(), CV_8UC1);

            if (!placed.mask.empty() &&
                placed.mask.cols == placed.patch.cols &&
                placed.mask.rows == placed.patch.rows)
            {
                cv::threshold(placed.mask, binary_mask, 128, 255, cv::THRESH_BINARY);
            }
            else if (!placed.hull.empty()) {

                std::vector<cv::Point> local_hull;

                for (const auto& p : placed.hull)
                {
                    int local_x = p.x - patch_left;
                    int local_y = p.y - patch_top;

                    if (local_x >= 0 && local_x < patch_part.cols &&
                        local_y >= 0 && local_y < patch_part.rows)
                    {
                        local_hull.emplace_back(local_x, local_y);
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

            int small_threshold = 10;

            bool isSmall = (patch_part.rows < small_threshold || patch_part.cols < small_threshold);

            if (!isSmall) {

                try {

                    cv::Mat result;

                    cv::Mat output_roi = output(target_bbox);

                    cv::Point local_center(
                        target_bbox.width / 2,
                        target_bbox.height / 2
                    );

                    local_center.x = std::max(0, std::min(local_center.x, target_bbox.width - 1));

                    local_center.y = std::max(0, std::min(local_center.y, target_bbox.height - 1));

                    if (local_center.x >= 0 && local_center.x < binary_mask.cols &&
                        local_center.y >= 0 && local_center.y < binary_mask.rows &&
                        binary_mask.at<uchar>(local_center.y, local_center.x) != 0) {
                        auto clone_mode = placed.scale_level == ScaleLevel::SMALL ? cv::MIXED_CLONE : cv::NORMAL_CLONE;

                        cv::seamlessClone(patch_part, output_roi, binary_mask, local_center, result, clone_mode);
                        result.copyTo(output(target_bbox));

                        groups_seamless++;
                    }
                    else {

                        GaussianCopy(binary_mask, patch_part, output, target_bbox);
                        groups_simple++;
                    }
                }
                catch (const cv::Exception& e) {

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
                        std::cout << "  patch_left: " << patch_left << ", patch_top: " << patch_top << std::endl;
                    }
                }
            }
            else {
                GaussianCopy(binary_mask, patch_part, output, target_bbox);
                groups_simple++;
                std::cout << i << " patch_part.rows=" << patch_part.rows << " patch_part.cols=" << patch_part.cols << std::endl;
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

        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
        std::cout << "Total pixel filling time: " << total_duration.count() / 1000.0 << " sec" << std::endl;

        std::cout << "FillPixels: copied " << groups_copied
            << " groups (seamless: " << groups_seamless
            << ", blended: " << groups_simple << ")" << std::endl;

        return output;
    }

}