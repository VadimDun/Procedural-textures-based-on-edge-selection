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

        // Заполняем фон средним цветом
        cv::Scalar avg_color = cv::mean(input_image);
        cv::Mat output = cv::Mat::zeros(size.width, size.width, input_image.type());
        output.setTo(avg_color);

        for (const auto& placed : placed_groups) {
            if (placed.source_index < 0 || placed.source_index >= source_infos.size()) {
                continue;
            }

            const auto& source_info = source_infos[placed.source_index];

            // Определяем bounding box для размещения
            cv::Rect target_bbox;
            if (!placed.hull.empty()) {
                target_bbox = cv::boundingRect(placed.hull);
            }
            else {
                target_bbox = placed.group.getBoundingBox();
            }

            // Корректируем границы
            if (target_bbox.x < 0) target_bbox.x = 0;
            if (target_bbox.y < 0) target_bbox.y = 0;
            if (target_bbox.x + target_bbox.width > size.width)
                target_bbox.width = size.width - target_bbox.x;
            if (target_bbox.y + target_bbox.height > size.height)
                target_bbox.height = size.height - target_bbox.y;

            if (target_bbox.width <= 0 || target_bbox.height <= 0) continue;

            // Берем патч из исходного изображения
            cv::Rect source_bbox = source_info.group.getBoundingBox();

            cv::Mat patch = input_image(source_bbox).clone();

            // Поворачиваем патч
            cv::Mat rotated_patch;
            if (std::abs(placed.rotation_angle) > 0.01f) {
                // Центр поворота — центр исходного bounding box
                cv::Point2f center(source_bbox.width / 2.0f, source_bbox.height / 2.0f);
                cv::Mat rot_mat = cv::getRotationMatrix2D(
                    center,
                    placed.rotation_angle * 180.0 / CV_PI,
                    placed.scale_factor
                );
                cv::warpAffine(patch, rotated_patch, rot_mat, patch.size());
            }
            else if (std::abs(placed.scale_factor - 1.0f) > 0.01f) {
                // Только масштабирование без поворота
                cv::resize(patch, rotated_patch, patch.size(),
                    placed.scale_factor, placed.scale_factor);
            }
            else {
                rotated_patch = patch;
            }

            // Масштабируем патч до размера target_bbox если нужно
            if (rotated_patch.cols != target_bbox.width || rotated_patch.rows != target_bbox.height) {
                cv::resize(rotated_patch, rotated_patch, target_bbox.size());
            }

            // Получаем маску
            cv::Mat mask_region;
            if (!placed.mask.empty()) {
                if (target_bbox.x >= 0 && target_bbox.y >= 0 &&
                    target_bbox.x + target_bbox.width <= placed.mask.cols &&
                    target_bbox.y + target_bbox.height <= placed.mask.rows) {
                    mask_region = placed.mask(target_bbox).clone();
                }
                else {
                    // Создаем маску из hull
                    if (!placed.hull.empty()) {
                        mask_region = cv::Mat::zeros(target_bbox.size(), CV_8UC1);
                        std::vector<std::vector<cv::Point>> hull_contour;
                        std::vector<cv::Point> shifted_hull = placed.hull;
                        for (auto& p : shifted_hull) {
                            p.x -= target_bbox.x;
                            p.y -= target_bbox.y;
                        }
                        hull_contour.push_back(shifted_hull);
                        cv::fillPoly(mask_region, hull_contour, cv::Scalar(255));
                    }
                    else {
                        mask_region = cv::Mat::ones(target_bbox.size(), CV_8UC1) * 255;
                    }
                }
            }
            else {
                // Создаем маску из hull
                if (!placed.hull.empty()) {
                    mask_region = cv::Mat::zeros(target_bbox.size(), CV_8UC1);
                    std::vector<std::vector<cv::Point>> hull_contour;
                    std::vector<cv::Point> shifted_hull = placed.hull;
                    for (auto& p : shifted_hull) {
                        p.x -= target_bbox.x;
                        p.y -= target_bbox.y;
                    }
                    hull_contour.push_back(shifted_hull);
                    cv::fillPoly(mask_region, hull_contour, cv::Scalar(255));
                }
                else {
                    mask_region = cv::Mat::ones(target_bbox.size(), CV_8UC1) * 255;
                }
            }

            // Масштабируем маску если нужно
            if (mask_region.cols != target_bbox.width || mask_region.rows != target_bbox.height) {
                cv::resize(mask_region, mask_region, target_bbox.size());
            }

            // Центр для seamlessClone
            cv::Point center(target_bbox.x + target_bbox.width / 2,
                target_bbox.y + target_bbox.height / 2);

            // Применяем Poisson blending
            cv::Mat result;

            bool use_seamless_clone = (rotated_patch.rows >= 10 && rotated_patch.cols >= 10);

            if (use_seamless_clone) {
                cv::seamlessClone(rotated_patch, output, mask_region, center, result, cv::NORMAL_CLONE);
                output = result; // Обновляем output
            }
            else {
                // Простое копирование для маленьких патчей
                cv::Rect roi(center.x - rotated_patch.cols / 2, center.y - rotated_patch.rows / 2,
                    rotated_patch.cols, rotated_patch.rows);

                // Убедимся, что ROI в пределах изображения
                roi = roi & cv::Rect(0, 0, output.cols, output.rows);

                if (roi.width > 0 && roi.height > 0) {
                    cv::Mat small_patch = rotated_patch(cv::Rect(0, 0, roi.width, roi.height));
                    small_patch.copyTo(output(roi));
                }
            }
        }

        return output;
    }

}