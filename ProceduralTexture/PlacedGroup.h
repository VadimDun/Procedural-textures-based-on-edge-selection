#pragma once

#include "EdgeGroup.h"
#include "SourceGroupInfo.h"
#include <opencv2/imgproc.hpp>

namespace EBPTns {

    struct PlacedGroup {
        cv::Mat patch;               // Трансформированный патч (уже повернут и масштабирован)
        cv::Mat mask;                // Трансформированная маска
        std::vector<cv::Point> hull; // Трансформированная выпуклая оболочка
        cv::Point2f position;
        short source_index;
        float rotation_angle;
        ScaleLevel scale_level;
        

        PlacedGroup() : source_index(-1), rotation_angle(0.0f), scale_level(ScaleLevel::MEDIUM) {}

        PlacedGroup(const cv::Mat& p, const cv::Mat& m, const std::vector<cv::Point>& h,
            const cv::Point2f& pos, short idx, float angle, ScaleLevel sl)
            : patch(p), mask(m), hull(h), position(pos),
            source_index(idx), rotation_angle(angle), scale_level(sl) {
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