#pragma once

#include <opencv2/opencv.hpp>
#include "EdgeGroup.h"
#include "PlacedGroup.h" 
#include <vector>
#include <random>

namespace EBPTns {

    class PixelSynthesis {
    public:
        PixelSynthesis();

        cv::Mat fillPixels(
            const cv::Mat& input_image,
            const std::vector<EdgeGroup>& source_groups,
            const std::vector<PlacedGroup>& placed_groups,
            int output_width, int output_height);

        //cv::Mat PatchCopy(
        //    const cv::Mat& input_image,
        //    const std::vector<EdgeGroup>& source_groups,
        //    const std::vector<PlacedGroup>& placed_groups,
        //    int output_width, int output_height);

        cv::Mat copyWithMask(const cv::Mat& source, const cv::Mat& mask,
            const cv::Rect& source_bbox, const cv::Rect& target_bbox);

        cv::Mat PatchCopy(
            const cv::Mat& input_image,
            const std::vector<SourceGroupInfo>& source_infos,
            const std::vector<PlacedGroup>& placed_groups,
            int output_width, int output_height);

        void setPatchSelectionMethod(int method) { patch_method_ = method; }
        void setRandomSeed(unsigned int seed);

    private:
        std::mt19937 rng_;

        // Метод выбора патчей
        int patch_method_ = 0;  // 0 = случайный, 1 = из центра группы

        cv::Rect getSourcePatchRect(const cv::Rect& target_bbox,
            const cv::Mat& input_image,
            const EdgeGroup* source_group = nullptr);
        void blendPatches(cv::Mat& output, const cv::Mat& patch,
            const cv::Rect& bbox, float alpha = 1.0f);
    };

} // namespace EBPTns