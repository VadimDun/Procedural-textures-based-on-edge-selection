#pragma once

#include <opencv2/opencv.hpp>
#include "Edge.h"
#include "EdgeGroup.h"
#include "PlacedGroup.h"
#include "TextureAnalysis.h"
#include <vector>
#include <random>

namespace EBPTns {

    class TextureSynthesis {
    public:

        struct ScaleLevelParams {
            float base_scale;        // Базовый масштаб патча
            float scale_variation;   // Вариация масштаба
            float angle_variation;   // Вариация масштаба
            float percent_fill_target; // Сколько процентов нужно заполнить

            ScaleLevelParams()
                : base_scale(1.0f), scale_variation(0.2f),
                angle_variation(0.0f), percent_fill_target(0.0f) {
            }
        };

        TextureSynthesis(const cv::Size& size, bool enable_rotation, bool enable_scaling);

        std::vector<PlacedGroup> synthesizeHierarchicalPlacement(
            const cv::Mat& input_image,
            const std::vector<Patch>& analysisResult);


		void setOutputSize(const cv::Size& size) { outputSize = size; }
        void setRandomSeed(unsigned int seed);
        void setTargetFillPercentage(float largePerc, float mediumPerc, float smallPerc) {
            scale_params_[ScaleLevel::LARGE].percent_fill_target = largePerc;
            scale_params_[ScaleLevel::MEDIUM].percent_fill_target = mediumPerc;
            scale_params_[ScaleLevel::SMALL].percent_fill_target = smallPerc;
		}

		cv::Size getOutputSize() const { return outputSize; }
		bool isRotationEnabled() const { return enable_rotation; }
		bool isScalingEnabled() const { return enable_scaling; }
        unsigned int getRandomSeed() const { return seed_; }

    private:
        std::mt19937 rng_;

        bool enable_rotation = true;
        bool enable_scaling = true;
        const int MIN_SIZE_PATCH = 300;
        cv::Size outputSize;
        unsigned int seed_ = 42;

        std::map<ScaleLevel, ScaleLevelParams> scale_params_;
        cv::Mat occupancy_map_;  // Карта заполнения

        void initScaleLevelParams();

        // Вспомогательные методы
        void updateOccupancyMap(const PlacedGroup& group);
        void erodeOccupancyMap(int width);
        float getOccupancyAtPoint(const cv::Point2f& point) const;

        cv::Point2f generatePositionForLarge();
        cv::Point findLargestEmptyLocation(float& radius);

        float generateRandomAngle(float variation);
        float generateRandomScale(float base_scale, float variation);

        PlacedGroup transformGroup(
            const Patch& patch,
            const cv::Mat& input_image,
            uint8_t source_idx,
            const cv::Point2f& position,
            float angle, float scale) const;
    };

}