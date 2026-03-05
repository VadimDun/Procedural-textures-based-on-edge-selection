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
        int output_width, int output_height) {

        cv::Mat output = cv::Mat::zeros(output_height, output_width, input_image.type());

        for (const auto& placed : placed_groups) {
            if (placed.source_index < 0 || placed.source_index >= source_infos.size()) {
                continue;
            }

            const auto& source_info = source_infos[placed.source_index];
            cv::Rect source_bbox = source_info.group.getBoundingBox();
            cv::Rect target_bbox = placed.group.getBoundingBox();

            if (target_bbox.x < 0) target_bbox.x = 0;
            if (target_bbox.y < 0) target_bbox.y = 0;
            if (target_bbox.x + target_bbox.width > output_width)
                target_bbox.width = output_width - target_bbox.x;
            if (target_bbox.y + target_bbox.height > output_height)
                target_bbox.height = output_height - target_bbox.y;

            if (target_bbox.width <= 0 || target_bbox.height <= 0) continue;

            cv::Mat mask = source_info.superpixel_mask;

            cv::Mat patch = copyWithMask(input_image, mask, source_bbox, target_bbox);

            // Масштабируем если нужно
            if (patch.cols != target_bbox.width || patch.rows != target_bbox.height) {
                cv::resize(patch, patch, target_bbox.size());
            }

            patch.copyTo(output(target_bbox));
        }

        return output;
    }

}