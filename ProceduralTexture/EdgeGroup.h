#pragma once

#include "Edge.h"
#include <vector>
#include <opencv2/core.hpp>

namespace EBPTns {

    class EdgeGroup {
    public:
        EdgeGroup() = default;
        EdgeGroup(const Edge& edge);
        EdgeGroup(const std::vector<Edge>& edges);

        void calculateStatistics();
        void addEdge(const Edge& edge);

        size_t getIndex() const { return index; }
        void setIndex(size_t idx) { index = idx; }

        const std::vector<Edge>& getEdges() const { return edges_; }
        cv::Point2f getCenter() const { return center_; }
        float getAverageAngle() const { return avg_angle_; }
        float getRadialSpread() const { return radial_spread_; }

        std::vector<cv::Point> getAllPoints() const;
        cv::Rect getBoundingBox() const;

        void translate(const cv::Point2f& offset);
        void rotate(float angle_radians);
        void scale(float factor);

    private:
        std::vector<Edge> edges_;    
        cv::Point2f center_;         
        float avg_angle_ = 0.0f;     
        float radial_spread_ = 0.0f; 
        size_t index;

        void calculateGroupCenter();
        void calculateAverageAngle();
        void calculateRadialSpread();
    };

}