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
            float density;           // Плотность размещения (0-2)
            float base_scale;        // Базовый масштаб патча
            float scale_variation;   // Вариация масштаба
            float angle_variation;   // Вариация масштаба
            int target_count;        // Целевое количество групп(todo добавление групп до определенного процента перекрытия)

            ScaleLevelParams()
                : density(1.0f), base_scale(1.0f), scale_variation(0.2f),
                target_count(0), angle_variation(0.0f) {
            }
        };

        //void setScaleLevelParams(ScaleLevel level, const ScaleLevelParams& params);
        void setScaleThresholds(float large_threshold, float medium_threshold, float small_threshold);

        // Новый метод для иерархического синтеза
        std::vector<PlacedGroup> synthesizeHierarchicalPlacement(
            const cv::Mat& input_image,
            const std::vector<SourceGroupInfo>& source_groups);


        TextureSynthesis(const cv::Size& size, bool enable_rotation);

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

        std::map<ScaleLevel, ScaleLevelParams> scale_params_;
        cv::Mat occupancy_map_;  // Карта заполнения

        // Пороги для классификации масштабов (в пикселях или радиусе)
        float large_scale_threshold_ = 100.0f;
        float medium_scale_threshold_ = 50.0f;
        float small_scale_threshold_ = 20.0f;

        // Вспомогательные методы
        void updateOccupancyMap(const PlacedGroup& group);
        float getOccupancyAtPoint(const cv::Point2f& point) const;
        cv::Point2f generatePositionByLevel(ScaleLevel level);
        bool checkOverlapByLevel(const PlacedGroup& new_group,
            const std::vector<PlacedGroup>& existing_groups,
            ScaleLevel current_level) const;
        void classifySourceGroups(std::vector<SourceGroupInfo>& source_groups);
        void checkAndAdjustThresholds(std::vector<SourceGroupInfo>& source_groups);
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