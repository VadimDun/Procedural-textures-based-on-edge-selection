// PlacedGroup.h
#pragma once

#include "EdgeGroup.h"
#include <opencv2/imgproc.hpp>

namespace EBPTns {

    struct PlacedGroup {
        cv::Mat patch;               // Трансформированный патч (уже повернут и масштабирован)
        cv::Mat mask;                // Трансформированная маска
        std::vector<cv::Point> hull; // Трансформированная выпуклая оболочка
        cv::Point2f position;
        int source_index;
        float scale_factor;
        float rotation_angle;

        PlacedGroup() : source_index(-1), scale_factor(1.0f), rotation_angle(0.0f) {}

        PlacedGroup(const cv::Mat& p, const cv::Mat& m, const std::vector<cv::Point>& h,
            const cv::Point2f& pos, int idx, float scale, float angle)
            : patch(p), mask(m), hull(h), position(pos),
            source_index(idx), scale_factor(scale), rotation_angle(angle) {
        }

        bool isValid() const { return !patch.empty() && source_index >= 0; }

        cv::Rect getBoundingBox() const {
            if (!hull.empty()) {
                return cv::boundingRect(hull);
            }
            return cv::Rect(0, 0, patch.cols, patch.rows);
        }
    };

}