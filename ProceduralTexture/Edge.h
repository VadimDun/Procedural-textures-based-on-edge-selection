#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace EBPTns {
    class Edge {
    public:
        Edge() = default;
        Edge(const std::vector<cv::Point>& points);
        Edge(const std::vector<cv::Point>& points, const std::vector<uchar>& chain);

        void calculateFeatures();
        void applyTransform(const cv::Mat& transform);

        //std::vector<cv::Point> reconstructFromChainCode(const std::vector<uchar>& chain, const cv::Point& start_point); // todo может пригодится в будущем

        const std::vector<cv::Point>& getPoints() const { return points_; }
        const std::vector<uchar>& getChainCode() const { return chain_code_; }
        cv::Point getStartPoint() const { return points_.empty() ? cv::Point(0, 0) : points_.front(); }

        float getLength() const { return length_; }
        cv::Point2f getCenter() const { return center_; }
        float getAngle() const { return angle_; }

        void setPoints(const std::vector<cv::Point>& points);

    private:
        std::vector<cv::Point> points_;       // Точки ребра
        std::vector<uchar> chain_code_;       // Цепной код Фримена (0-7)
        cv::Point start_point_;               // Начальная точка для цепного кода

        float length_ = 0.0f;
        cv::Point2f center_;
        float angle_ = 0.0f;

        void calculateLength();
        void calculateCenter();
        void calculateAngle();

        void pointsToChainCode();
        void chainCodeToPoints();

        static uchar getDirection(const cv::Point& from, const cv::Point& to);
        static cv::Point getOffset(uchar direction);
    };

}