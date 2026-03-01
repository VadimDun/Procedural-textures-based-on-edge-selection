#include "EdgeGroup.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>

namespace EBPTns {

    EdgeGroup::EdgeGroup(const std::vector<Edge>& edges)
        : edges_(edges) {
        calculateStatistics();
    }

    void EdgeGroup::calculateStatistics() {
        if (edges_.empty()) {
            center_ = cv::Point2f(0, 0);
            avg_angle_ = 0.0f;
            radial_spread_ = 0.0f;
            return;
        }

        calculateGroupCenter();
        calculateAverageAngle();
        calculateRadialSpread();
    }

    void EdgeGroup::calculateGroupCenter() {
        if (edges_.empty()) {
            center_ = cv::Point2f(0, 0);
            return;
        }

        float sum_x = 0.0f, sum_y = 0.0f;

        for (const auto& edge : edges_) {
            cv::Point2f edge_center = edge.getCenter();
            sum_x += edge_center.x;
            sum_y += edge_center.y;
        }

        center_.x = sum_x / static_cast<float>(edges_.size());
        center_.y = sum_y / static_cast<float>(edges_.size());
    }

    void EdgeGroup::calculateAverageAngle() {
        if (edges_.empty()) {
            avg_angle_ = 0.0f;
            return;
        }

        float sum_cos = 0.0f, sum_sin = 0.0f;

        for (const auto& edge : edges_) {
            float angle = edge.getAngle();
            sum_cos += std::cos(angle);
            sum_sin += std::sin(angle);
        }

        avg_angle_ = std::atan2(sum_sin, sum_cos);

        if (avg_angle_ < 0) {
            avg_angle_ += CV_PI;
        }
    }

    void EdgeGroup::calculateRadialSpread() {
        if (edges_.empty()) {
            radial_spread_ = 0.0f;
            return;
        }

        float max_distance = 0.0f;

        for (const auto& edge : edges_) {
            cv::Point2f edge_center = edge.getCenter();
            float dx = edge_center.x - center_.x;
            float dy = edge_center.y - center_.y;
            float distance = std::sqrt(dx * dx + dy * dy);

            if (distance > max_distance) {
                max_distance = distance;
            }
        }

        radial_spread_ = max_distance;
    }

    void EdgeGroup::addEdge(const Edge& edge) {
        edges_.push_back(edge);
        calculateStatistics();
    }

    std::vector<cv::Point> EdgeGroup::getAllPoints() const {
        std::vector<cv::Point> all_points;

        for (const auto& edge : edges_) {
            const auto& edge_points = edge.getPoints();
            all_points.insert(all_points.end(),
                edge_points.begin(),
                edge_points.end());
        }

        return all_points;
    }

    cv::Rect EdgeGroup::getBoundingBox() const {
        if (edges_.empty()) {
            return cv::Rect(0, 0, 0, 0);
        }

        auto all_points = getAllPoints();
        if (all_points.empty()) {
            return cv::Rect(0, 0, 0, 0);
        }

        return cv::boundingRect(all_points);
    }

    void EdgeGroup::translate(const cv::Point2f& offset) {
        cv::Mat translation_matrix = (cv::Mat_<float>(2, 3) <<
            1, 0, offset.x,
            0, 1, offset.y);

        for (auto& edge : edges_) {
            edge.applyTransform(translation_matrix);
        }

        center_.x += offset.x;
        center_.y += offset.y;
    }

    void EdgeGroup::rotate(float angle_radians) {
        if (edges_.empty()) return;

        cv::Mat rotation_matrix = cv::getRotationMatrix2D(center_,
            angle_radians * 180.0f / CV_PI,
            1.0);

        for (auto& edge : edges_) {
            edge.applyTransform(rotation_matrix);
        }

        avg_angle_ = std::fmod(avg_angle_ + angle_radians, CV_PI);
        if (avg_angle_ < 0) {
            avg_angle_ += CV_PI;
        }
    }

    void EdgeGroup::scale(float factor) {
        if (edges_.empty() || factor <= 0.0f) return;

        cv::Mat scale_matrix = (cv::Mat_<float>(2, 3) <<
            factor, 0, center_.x * (1 - factor),
            0, factor, center_.y * (1 - factor));

        for (auto& edge : edges_) {
            edge.applyTransform(scale_matrix);
        }

        radial_spread_ *= factor;
    }

}