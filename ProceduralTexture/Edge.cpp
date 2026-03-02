#include "Edge.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>

namespace EBPTns {

    Edge::Edge(const std::vector<cv::Point>& points) : points_(points) {
        pointsToChainCode();
        calculateFeatures();
    }

    Edge::Edge(const std::vector<cv::Point>& points, const std::vector<uchar>& chain)
        : points_(points), chain_code_(chain) {
        if (!points_.empty()) {
            start_point_ = points_.front();
        }
        calculateFeatures();
    }

    void Edge::pointsToChainCode() {
        chain_code_.clear();
        if (points_.size() < 2) {
            if (!points_.empty()) {
                start_point_ = points_.front();
            }
            return;
        }

        chain_code_.reserve(points_.size() - 1);
        start_point_ = points_.front();

        for (size_t i = 1; i < points_.size(); ++i) {
            uchar dir = getDirection(points_[i - 1], points_[i]);
            chain_code_.push_back(dir);
        }
    }

    void Edge::chainCodeToPoints() {
        points_.clear();
        if (chain_code_.empty()) {
            if (!points_.empty()) {
                points_ = { start_point_ };
            }
            return;
        }

        points_.reserve(chain_code_.size() + 1);
        points_.push_back(start_point_);

        cv::Point current = start_point_;
        for (uchar dir : chain_code_) {
            cv::Point offset = getOffset(dir);
            current.x += offset.x;
            current.y += offset.y;
            points_.push_back(current);
        }
    }

    // Получаем направление цепного кода
    uchar Edge::getDirection(const cv::Point& from, const cv::Point& to) {
        int dx = to.x - from.x;
        int dy = to.y - from.y;

        // Нормализуем
        dx = (dx > 0) ? 1 : (dx < 0) ? -1 : 0;
        dy = (dy > 0) ? 1 : (dy < 0) ? -1 : 0;

        if (dx == 1 && dy == 0) return 0;   // вправо
        if (dx == 1 && dy == -1) return 1;  // вправо-вверх
        if (dx == 0 && dy == -1) return 2;  // вверх
        if (dx == -1 && dy == -1) return 3; // влево-вверх
        if (dx == -1 && dy == 0) return 4;  // влево
        if (dx == -1 && dy == 1) return 5;  // влево-вниз
        if (dx == 0 && dy == 1) return 6;   // вниз
        if (dx == 1 && dy == 1) return 7;   // вправо-вниз

        return 0;
    }

    // Получаем смещение при обратном кодировании из цепного кода
    cv::Point Edge::getOffset(uchar direction) {
        switch (direction) {
        case 0: return cv::Point(1, 0);   // вправо
        case 1: return cv::Point(1, -1);  // вправо-вверх
        case 2: return cv::Point(0, -1);  // вверх
        case 3: return cv::Point(-1, -1); // влево-вверх
        case 4: return cv::Point(-1, 0);  // влево
        case 5: return cv::Point(-1, 1);  // влево-вниз
        case 6: return cv::Point(0, 1);   // вниз
        case 7: return cv::Point(1, 1);   // вправо-вниз
        default: return cv::Point(0, 0);
        }
    }

    void Edge::calculateFeatures() {
        if (points_.empty()) return;

        calculateLength();
        calculateCenter();
        calculateAngle();
    }

    void Edge::calculateLength() {
        length_ = static_cast<float>(chain_code_.size() + 1);
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

        cv::Mat data(points_.size(), 2, CV_32F);
        for (size_t i = 0; i < points_.size(); ++i) {
            data.at<float>(i, 0) = static_cast<float>(points_[i].x);
            data.at<float>(i, 1) = static_cast<float>(points_[i].y);
        }

        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

        // Первый собственный вектор - направление наибольшей дисперсии (вдоль ребра)
        cv::Mat eigenvector = pca.eigenvectors.row(0);
        float x = eigenvector.at<float>(0, 0);
        float y = eigenvector.at<float>(0, 1);

        angle_ = std::atan2(y, x);

        // Нормализуем
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

        pointsToChainCode();
        calculateFeatures();
    }

    void Edge::setPoints(const std::vector<cv::Point>& points) {
        points_ = points;
        pointsToChainCode();
        calculateFeatures();
    }

}