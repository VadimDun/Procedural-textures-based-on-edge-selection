#include "TextureAnalysis.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace EBPTns {

    TextureAnalysis::TextureAnalysis() {
        setCannyThresholds(50, 150);
        setMinEdgeLength(10);
        setGroupingDistance(50);
    }

    void TextureAnalysis::setCannyThresholds(double low, double high) {
        canny_low_threshold_ = low;
        canny_high_threshold_ = high;
    }

    void TextureAnalysis::setMinEdgeLength(int min_length) {
        min_edge_length_ = min_length;
    }

    void TextureAnalysis::setGroupingDistance(int distance) {
        grouping_distance_ = distance;
    }

    EBPT TextureAnalysis::analyzeTexture(const cv::Mat& input_image) {
        std::vector<Edge> edges = extractEdges(input_image);
        if (edges.empty()) {
            return EBPT(input_image);
        }
        std::vector<EdgeGroup> groups = groupEdges(edges);
        if (groups.empty()) {
            return EBPT(input_image);
        }
        EBPT ebpt_model(input_image);
        for (const auto& group : groups) {
            ebpt_model.addEdgeGroup(group);
        }
        return ebpt_model;
    }

    std::vector<Edge> TextureAnalysis::extractEdges(const cv::Mat& image) {
        std::vector<Edge> edges;
        if (image.empty()) {
            return edges;
        }
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        else {
            gray = image.clone();
        }
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);
        cv::Mat edges_image;
        cv::Canny(blurred, edges_image,
            canny_low_threshold_,
            canny_high_threshold_);
        auto contours = findContours(edges_image);
        for (const auto& contour : contours) {
            if (contour.size() < 2) continue;
            auto& simplified = contour;
            float length = 0;
            for (size_t i = 1; i < simplified.size(); ++i) {
                float dx = simplified[i].x - simplified[i - 1].x;
                float dy = simplified[i].y - simplified[i - 1].y;
                length += std::sqrt(dx * dx + dy * dy);
            }
            if (length < min_edge_length_) {
                continue;
            }
            Edge edge(simplified);
            edges.push_back(edge);
        }
        std::sort(edges.begin(), edges.end(),
            [](const Edge& a, const Edge& b) {
                return a.getLength() > b.getLength();
            });
        return edges;
    }

    std::vector<std::vector<cv::Point>> TextureAnalysis::findContours(const cv::Mat& edges_image) {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(edges_image, contours, hierarchy,
            cv::RETR_LIST,
            cv::CHAIN_APPROX_SIMPLE);
        return contours;
    }

    std::vector<cv::Point> TextureAnalysis::simplifyContour(const std::vector<cv::Point>& contour) {
        if (contour.size() < 3) {
            return contour;
        }
        std::vector<cv::Point> simplified;
        for (size_t i = 0; i < contour.size(); i += 3) {
            simplified.push_back(contour[i]);
        }
        if (simplified.empty() ||
            simplified.back() != contour.back()) {
            simplified.push_back(contour.back());
        }
        return simplified;
    }

    bool TextureAnalysis::shouldGroup(const Edge& edge1, const Edge& edge2) const {
        cv::Point2f center1 = edge1.getCenter();
        cv::Point2f center2 = edge2.getCenter();
        float dx = center1.x - center2.x;
        float dy = center1.y - center2.y;
        float distance = std::sqrt(dx * dx + dy * dy);
        if (distance > grouping_distance_) {
            return false;
        }
        float angle1 = edge1.getAngle();
        float angle2 = edge2.getAngle();
        float angle_diff = std::abs(angle1 - angle2);
        if (angle_diff > CV_PI / 2) {
            angle_diff = CV_PI - angle_diff;
        }
        return true;
    }

    std::vector<EdgeGroup> TextureAnalysis::groupEdges(const std::vector<Edge>& edges) {
        std::vector<EdgeGroup> groups;
        if (edges.empty()) {
            return groups;
        }
        std::vector<bool> assigned(edges.size(), false);
        for (size_t i = 0; i < edges.size(); ++i) {
            if (assigned[i]) continue;
            std::vector<Edge> group_edges;
            group_edges.push_back(edges[i]);
            assigned[i] = true;
            bool found_more;
            do {
                found_more = false;
                for (size_t j = i + 1; j < edges.size(); ++j) {
                    if (assigned[j]) continue;
                    bool fits_to_group = false;
                    for (const auto& group_edge : group_edges) {
                        if (shouldGroup(group_edge, edges[j])) {
                            fits_to_group = true;
                            break;
                        }
                    }
                    if (fits_to_group) {
                        group_edges.push_back(edges[j]);
                        assigned[j] = true;
                        found_more = true;
                    }
                }
            } while (found_more);
            if (group_edges.size() >= 1) {
                EdgeGroup group(group_edges);
                groups.push_back(group);
            }
        }
        std::sort(groups.begin(), groups.end(),
            [](const EdgeGroup& a, const EdgeGroup& b) {
                return a.getRadialSpread() > b.getRadialSpread();
            });
        return groups;
    }

    cv::Mat TextureAnalysis::visualizeEdges(const cv::Mat& image,
        const std::vector<Edge>& edges) {
        cv::Mat visualization;
        if (image.channels() == 1) {
            cv::cvtColor(image, visualization, cv::COLOR_GRAY2BGR);
        }
        else {
            visualization = image.clone();
        }
        for (const auto& edge : edges) {
            const auto& points = edge.getPoints();
            if (points.size() < 2) continue;
            for (size_t i = 1; i < points.size(); ++i) {
                cv::line(visualization, points[i - 1], points[i],
                    cv::Scalar(0, 255, 0),
                    2);
            }
            cv::Point center = cv::Point(
                static_cast<int>(edge.getCenter().x),
                static_cast<int>(edge.getCenter().y)
            );
            cv::circle(visualization, center, 3,
                cv::Scalar(0, 0, 255), -1);
        }
        return visualization;
    }

    cv::Mat TextureAnalysis::visualizeGroups(const cv::Mat& image,
        const std::vector<EdgeGroup>& groups) {
        cv::Mat visualization;
        if (image.channels() == 1) {
            cv::cvtColor(image, visualization, cv::COLOR_GRAY2BGR);
        }
        else {
            visualization = image.clone();
        }
        std::vector<cv::Scalar> group_colors = {
            cv::Scalar(255, 0, 0),
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 0, 255),
            cv::Scalar(255, 255, 0),
            cv::Scalar(255, 0, 255),
            cv::Scalar(0, 255, 255),
            cv::Scalar(128, 0, 0),
            cv::Scalar(0, 128, 0),
            cv::Scalar(0, 0, 128),
            cv::Scalar(128, 128, 0)
        };
        for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
            const auto& group = groups[group_idx];
            cv::Scalar color = group_colors[group_idx % group_colors.size()];
            for (const auto& edge : group.getEdges()) {
                const auto& points = edge.getPoints();
                if (points.size() < 2) continue;
                for (size_t i = 1; i < points.size(); ++i) {
                    cv::line(visualization, points[i - 1], points[i],
                        color,
                        2);
                }
            }
            cv::Point group_center = cv::Point(
                static_cast<int>(group.getCenter().x),
                static_cast<int>(group.getCenter().y)
            );
            cv::circle(visualization, group_center, 5,
                color, -1);
            cv::Rect bbox = group.getBoundingBox();
            cv::rectangle(visualization, bbox, color, 2);
            std::string label = "G" + std::to_string(group_idx + 1);
            cv::putText(visualization, label,
                cv::Point(bbox.x, bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1);
        }
        return visualization;
    }
}