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

        std::vector<PlacedGroup> synthesizeHierarchicalPlacement(
            const cv::Mat& input_image,
            const std::vector<SourceGroupInfo>& source_groups);


        TextureSynthesis(const cv::Size& size, bool enable_rotation);

        void setRandomSeed(unsigned int seed);
        void setAvoidOverlap(bool avoid) { avoid_overlap_ = avoid; }
        void setMinDistance(float distance) { min_distance_ = distance; }

    private:
        std::mt19937 rng_;

        bool avoid_overlap_ = true;
        float min_distance_ = 30.0f;
        cv::Size outputSize;

        std::map<ScaleLevel, ScaleLevelParams> scale_params_;
        cv::Mat occupancy_map_;  // Карта заполнения

        // Вспомогательные методы
        void updateOccupancyMap(const PlacedGroup& group);
        float getOccupancyAtPoint(const cv::Point2f& point) const;
        cv::Point2f generatePositionByLevel(ScaleLevel level);
        bool checkOverlapByLevel(const PlacedGroup& new_group,
            const std::vector<PlacedGroup>& existing_groups,
            ScaleLevel current_level) const;
        void initScaleLevelParams(bool enable_rotation);
        float computeHullIntersectionArea(
            const std::vector<cv::Point>& hull1,
            const std::vector<cv::Point>& hull2) const;


        cv::Point2f generateRandomPosition();
        float generateRandomAngle(float variation);
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

        bool checkHullIntersection(const std::vector<cv::Point>& hull1,
            const std::vector<cv::Point>& hull2) const;
        bool doSegmentsIntersect(const cv::Point& p1, const cv::Point& p2,
            const cv::Point& q1, const cv::Point& q2) const;
        bool onSegment(const cv::Point& p, const cv::Point& q, const cv::Point& r) const;
        bool isPointInPolygon(const cv::Point& point, const std::vector<cv::Point>& polygon) const;
    };

}