#include "Edge.h"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace EBPTns {

    Edge::Edge(const std::vector<cv::Point>& points)
        : points_(points) {
        calculateFeatures();
    }

    void Edge::calculateFeatures() {
        if (points_.empty()) return;
        calculateLength();
        calculateCenter();
        calculateAngle();
    }

    void Edge::calculateLength() {
        length_ = 0.0f;
        for (size_t i = 1; i < points_.size(); ++i) {
            float dx = static_cast<float>(points_[i].x - points_[i - 1].x);
            float dy = static_cast<float>(points_[i].y - points_[i - 1].y);
            length_ += std::sqrt(dx * dx + dy * dy);
        }
        if (length_ == 0.0f && !points_.empty()) {
            length_ = 1.0f;
        }
    }

    void Edge::calculateCenter() {
        if (points_.empty()) {
            center_ = cv::Point2f(0, 0);
            return;
        }
        float sum_x = 0.0f, sum_y = 0.0f;
        for (const auto& point : points_) {
            sum_x += static_cast<float>(point.x);
            sum_y += static_cast<float>(point.y);
        }
        center_.x = sum_x / static_cast<float>(points_.size());
        center_.y = sum_y / static_cast<float>(points_.size());
    }

    void Edge::calculateAngle() {
        if (points_.size() < 2) {
            angle_ = 0.0f;
            return;
        }
        cv::Point start = points_.front();
        cv::Point end = points_.back();
        float dx = static_cast<float>(end.x - start.x);
        float dy = static_cast<float>(end.y - start.y);
        angle_ = std::atan2(dy, dx);
        if (angle_ < 0) {
            angle_ += CV_PI;
        }
    }

    void Edge::applyTransform(const cv::Mat& transform) {
        if (points_.empty() || transform.empty()) return;
        std::vector<cv::Point2f> points2f;
        points2f.reserve(points_.size());
        for (const auto& p : points_) {
            points2f.emplace_back(static_cast<float>(p.x),
                static_cast<float>(p.y));
        }
        cv::transform(points2f, points2f, transform);
        points_.clear();
        points_.reserve(points2f.size());
        for (const auto& p : points2f) {
            points_.emplace_back(static_cast<int>(p.x),
                static_cast<int>(p.y));
        }
        calculateFeatures();
    }

    void Edge::setPoints(const std::vector<cv::Point>& points) {
        points_ = points;
        calculateFeatures();
    }

}