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
        TextureSynthesis();

        std::vector<PlacedGroup> synthesizePlacement(
            const std::vector<EdgeGroup>& source_groups,
            int output_width, int output_height,
            float density = 0.7f,
            float angle_variation = 0.3f,
            float scale_variation = 0.2f);

        std::vector<PlacedGroup> synthesizeFromEBPT(
            const EBPT& ebpt_model,
            int output_width, int output_height);

        void setRandomSeed(unsigned int seed);
        void setAvoidOverlap(bool avoid) { avoid_overlap_ = avoid; }
        void setMinDistance(float distance) { min_distance_ = distance; }

        cv::Mat drawPlacementMap(
            const std::vector<PlacedGroup>& placed_groups,
            int width, int height);

    private:
        std::mt19937 rng_;

        bool avoid_overlap_ = true;
        float min_distance_ = 30.0f;

        cv::Point2f generateRandomPosition(int width, int height);
        float generateRandomAngle(float base_angle, float variation);
        float generateRandomScale(float base_scale, float variation);

        bool checkOverlap(const EdgeGroup& group1,
            const EdgeGroup& group2,
            float min_distance);

        PlacedGroup transformGroup(const EdgeGroup& source_group,
            int source_idx,
            const cv::Point2f& position,
            float angle, float scale);
    };

}