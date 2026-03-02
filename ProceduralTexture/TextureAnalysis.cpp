#include "TextureAnalysis.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

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

    // Structured forests
    bool TextureAnalysis::initializeStructuredDetector(const std::string& model_path) {
        // Существует ли файл модели
        std::ifstream model_file(model_path);
        if (!model_file.good()) {
            std::cerr << "TextureAnalysis::initializeStructuredDetector: Model isn't found at path: " << model_path << std::endl;
            is_structured_initialized_ = false;
            return false;
        }

        try {
            structured_edge_detector_ = cv::ximgproc::createStructuredEdgeDetection(model_path);
            is_structured_initialized_ = true;
            std::cout << "StructuredEdgeDetection inited" << std::endl;
            return true;
        }
        catch (const cv::Exception& e) {
            std::cerr << "TextureAnalysis::initializeStructuredDetector: StructuredEdgeDetection didn't created: " << e.what() << std::endl;
            is_structured_initialized_ = false;
            return false;
        }
    }

    std::vector<Edge> TextureAnalysis::extractEdgesStructured(const cv::Mat& image, cv::Mat edge_probability_map) {
        std::vector<Edge> edges;

        if (image.empty()) {
            std::cerr << "Empty image" << std::endl;
            return edges;
        }

        if (!is_structured_initialized_) {
            std::cerr << "TextureAnalysis::extractEdgesStructured: StructuredEdgeDetection didn't created "
                << "Firstly call analyzeTextureStructured or initializeStructuredDetector" << std::endl;
            return edges;
        }

        // Бинаризуем карту вероятностей (порог можно подобрать)
        cv::Mat binary_edges;
        double threshold = 0.25; // todo сделать изменение порога в ручном формате в процессе работы программы, либо автоматический подбор
        cv::threshold(edge_probability_map, binary_edges, threshold, 255, cv::THRESH_BINARY);
        binary_edges.convertTo(binary_edges, CV_8UC1);

        auto contours = findContours(binary_edges);

        // Фильтруем и создаем объекты Edge
        for (const auto& contour : contours) {
            if (contour.size() < 2) continue;

            auto simplified = simplifyContour(contour);

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

        return edges;
    }

    AnalysisResult TextureAnalysis::analyzeTextureStructured(
        const cv::Mat& input_image,
        const std::string& model_path) {

        std::cout << "Picture size: " << input_image.cols << "x" << input_image.rows << std::endl;

        AnalysisResult empty_result;

        if (!initializeStructuredDetector(model_path)) {
            std::cerr << "TextureAnalysis::analyzeTextureStructured: StructuredForests didn't created" << std::endl;
            return empty_result;
        }

        // Конвертируем в float и нормализуем
        cv::Mat float_image;
        if (input_image.channels() == 3) {
            cv::cvtColor(input_image, float_image, cv::COLOR_BGR2RGB); // Модель обучена на RGB
        }
        else {
            cv::cvtColor(input_image, float_image, cv::COLOR_GRAY2RGB); // Если grayscale, делаем 3 канала
        }
        float_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

        cv::Mat edge_probability_map;
        structured_edge_detector_->detectEdges(float_image, edge_probability_map);

        // Получаем рёбра
        std::vector<Edge> edges = extractEdgesStructured(input_image, edge_probability_map);
        if (edges.empty()) {
            std::cerr << "Any edge didn't found" << std::endl;
            return empty_result;
        }

        std::vector<EdgeGroup> groups = groupEdges(edges);
        if (groups.empty()) {
            std::cerr << "Any group didn't created" << std::endl;
            return empty_result;
        }

        // Создаем EBPT
        EBPT ebpt_model(input_image);
        for (const auto& group : groups) {
            ebpt_model.addEdgeGroup(group);
        }

        // Создаем визуализации для отладки. Todo потом может удалить
        cv::Mat edges_visualization = visualizeEdges(input_image, edges);
        cv::Mat groups_visualization = visualizeGroups(input_image, groups);

        std::cout << "   Edges found: " << edges.size() << std::endl;
        std::cout << "   Groups created: " << groups.size() << std::endl;

        return AnalysisResult(ebpt_model, edges, groups,
            edges_visualization, groups_visualization,
            edge_probability_map);
    }

    AnalysisResult TextureAnalysis::analyzeTexture(const cv::Mat& input_image) {
        AnalysisResult result;

        result.edges = extractEdges(input_image);
        result.edges_visualization = visualizeEdges(input_image, result.edges);

        result.groups = groupEdges(result.edges);
        result.groups_visualization = visualizeGroups(input_image, result.groups);

        result.modelEBPT = EBPT(input_image);
        for (const auto& group : result.groups) {
            result.modelEBPT.addEdgeGroup(group);
        }

        return result;
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