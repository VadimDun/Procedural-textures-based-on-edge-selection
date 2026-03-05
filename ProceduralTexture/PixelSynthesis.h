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
        PixelSynthesis();

        cv::Mat copyWithMask(const cv::Mat& source, const cv::Mat& mask,
            const cv::Rect& source_bbox, const cv::Rect& target_bbox);

        cv::Mat fillPixels(
            const cv::Mat& input_image,
            const std::vector<SourceGroupInfo>& source_infos,
            const std::vector<PlacedGroup>& placed_groups,
            int output_width, int output_height);

        void setRandomSeed(unsigned int seed);

    private:
        std::mt19937 rng_;

        void blendPatches(cv::Mat& output, const cv::Mat& patch,
            const cv::Rect& bbox, float alpha = 1.0f);
    };

}