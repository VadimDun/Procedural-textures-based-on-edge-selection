#pragma once

#include <opencv2/opencv.hpp>
#include "Edge.h"
#include "EdgeGroup.h"
#include "PlacedGroup.h"
#include "EBPT.h"
#include <vector>
#include <random>

namespace EBPTns {



    class TextureSynthesis {
    public:
        TextureSynthesis(const cv::Size& size);

        std::vector<PlacedGroup> synthesizePlacement(
            const cv::Mat& input_image,
            const std::vector<SourceGroupInfo>& source_groups,
            float density,
            float angle_variation,
            float scale_variation);

        //std::vector<PlacedGroup> synthesizeFromEBPT(
        //    const EBPT& ebpt_model,
        //    int output_width, int output_height);

        void setRandomSeed(unsigned int seed);
        void setAvoidOverlap(bool avoid) { avoid_overlap_ = avoid; }
        void setMinDistance(float distance) { min_distance_ = distance; }

    private:
        std::mt19937 rng_;

        bool avoid_overlap_ = true;
        float min_distance_ = 30.0f;
        cv::Size outputSize;

        cv::Point2f generateRandomPosition();
        float generateRandomAngle(float base_angle, float variation);
        float generateRandomScale(float base_scale, float variation);

        bool checkOverlap(const EdgeGroup& group1,
            const EdgeGroup& group2,
            float min_distance);

        PlacedGroup transformGroup(
            const SourceGroupInfo& source_info,
            const cv::Mat& input_image,
            int source_idx,
            const cv::Point2f& position,
            float angle, float scale) const;
    };

}