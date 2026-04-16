#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include "EdgeGroup.h"
#include "PlacedGroup.h" 
#include "EBPT.h"

namespace EBPTns {

    class PixelSynthesis {
    public:

        cv::Mat fillPixels(
            const cv::Mat& input_image,
            const std::vector<SourceGroupInfo>& source_infos,
            const std::vector<PlacedGroup>& placed_groups,
            const cv::Size& size);

    private:
        std::mt19937 rng_;

        void GaussianCopy(const cv::Mat& binary_mask, const cv::Mat& patch_part, cv::Mat& output, const cv::Rect& target_bbox);
    };

}