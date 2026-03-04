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

        std::sort(edges.begin(), edges.end(),
            [](const auto& a, const auto& b) {
                return a.getLength() > b.getLength();
            });
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
            if (contour.size() < min_edge_length_) continue;
            //auto& simplified = contour;
            //float length = 0;
            //for (size_t i = 1; i < simplified.size(); ++i) {
            //    float dx = simplified[i].x - simplified[i - 1].x;
            //    float dy = simplified[i].y - simplified[i - 1].y;
            //    length += std::sqrt(dx * dx + dy * dy);
            //}
            //if (length < min_edge_length_) {
            //    continue;
            //}
            Edge edge(contour);
            //if (edge.getLength() < min_edge_length_) continue;
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

    /////////////////////////////
    // Суперпиксели
    /////////////////////////////
    void TextureAnalysis::setSuperpixelParams(int region_size, float ruler) {
        superpixel_region_size_ = region_size;
        superpixel_ruler_ = ruler;
        std::cout << "Параметры суперпикселей: region_size=" << region_size
            << ", ruler=" << ruler << std::endl;
    }

    cv::Mat TextureAnalysis::computeSuperpixels(const cv::Mat& image, int region_size, float ruler) {

        // Конвертируем в LAB цветовое пространство (лучше для суперпикселей)
        cv::Mat lab_image;
        if (image.channels() == 3) {
            cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);
        }
        else {
            cv::cvtColor(image, lab_image, cv::COLOR_GRAY2BGR);
            cv::cvtColor(lab_image, lab_image, cv::COLOR_BGR2Lab);
        }

        cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic =
            cv::ximgproc::createSuperpixelSLIC(lab_image,
                cv::ximgproc::SLIC,
                region_size,
                ruler);

        slic->iterate(20);

        cv::Mat labels;
        slic->getLabels(labels);

        int num_superpixels = slic->getNumberOfSuperpixels();
        std::cout << "   Superpixels created: " << num_superpixels << std::endl;

        return labels;
    }

    std::unordered_map<int, std::vector<Edge>> TextureAnalysis::assignEdgesToSuperpixels(
        const std::vector<Edge>& edges, const cv::Mat& labels) {

        std::unordered_map<int, std::vector<Edge>> superpixel_map;

        if (labels.empty()) {
            std::cerr << "labels is empty" << std::endl;
            return superpixel_map;
        }

        for (const auto& edge : edges) {
            cv::Point2f center = edge.getCenter();
            int x = static_cast<int>(center.x);
            int y = static_cast<int>(center.y);

            if (x < 0 || x >= labels.cols || y < 0 || y >= labels.rows) {
                continue;
            }

            int sp_id = labels.at<int>(y, x);
            superpixel_map[sp_id].push_back(edge);
        }

        std::cout << "   Superpixels with edges " << superpixel_map.size() << std::endl;

        // Подсчет суперпикселей с разным количеством ребер
        int empty_sp = 0;
        int sp_with_1 = 0;
        int sp_with_2_5 = 0;
        int sp_with_6_plus = 0;

        for (const auto& [sp_id, sp_edges] : superpixel_map) {
            size_t count = sp_edges.size();
            if (count == 0) empty_sp++;
            else if (count == 1) sp_with_1++;
            else if (count <= 5) sp_with_2_5++;
            else sp_with_6_plus++;
        }

        std::cout << "   Статистика суперпикселей:" << std::endl;
        std::cout << "     - Пустых: " << empty_sp << std::endl;
        std::cout << "     - С 1 ребром: " << sp_with_1 << std::endl;
        std::cout << "     - С 2-5 ребрами: " << sp_with_2_5 << std::endl;
        std::cout << "     - С 6+ ребрами: " << sp_with_6_plus << std::endl;

        return superpixel_map;
    }

    cv::Mat TextureAnalysis::visualizeSuperpixels(const cv::Mat& image, const cv::Mat& labels) {
        cv::Mat visualization;

        if (image.channels() == 1) {
            cv::cvtColor(image, visualization, cv::COLOR_GRAY2BGR);
        }
        else {
            visualization = image.clone();
        }

        if (labels.empty()) {
            return visualization;
        }

        std::unordered_map<int, cv::Scalar> sp_colors;
        cv::RNG rng(12345);

        // Рисуем границы суперпикселей
        cv::Mat contour_mask = cv::Mat::zeros(labels.size(), CV_8UC1);

        // Проходим по всем пикселям и находим границы
        for (int y = 1; y < labels.rows - 1; ++y) {
            for (int x = 1; x < labels.cols - 1; ++x) {
                int current = labels.at<int>(y, x);

                // Проверяем соседей
                if (current != labels.at<int>(y - 1, x) ||
                    current != labels.at<int>(y + 1, x) ||
                    current != labels.at<int>(y, x - 1) ||
                    current != labels.at<int>(y, x + 1)) {
                    contour_mask.at<uchar>(y, x) = 255;
                }
            }
        }

        // Рисуем границы красным
        visualization.setTo(cv::Scalar(0, 0, 255), contour_mask);

        // Добавляем центры суперпикселей
        std::unordered_map<int, cv::Point> centers;
        std::unordered_map<int, int> counts;

        for (int y = 0; y < labels.rows; ++y) {
            for (int x = 0; x < labels.cols; ++x) {
                int sp_id = labels.at<int>(y, x);
                centers[sp_id].x += x;
                centers[sp_id].y += y;
                counts[sp_id]++;
            }
        }

        for (auto& [sp_id, center] : centers) {
            if (counts[sp_id] > 0) {
                center.x /= counts[sp_id];
                center.y /= counts[sp_id];

                if (sp_colors.find(sp_id) == sp_colors.end()) {
                    sp_colors[sp_id] = cv::Scalar(rng.uniform(0, 255),
                        rng.uniform(0, 255),
                        rng.uniform(0, 255));
                }

                cv::circle(visualization, center, 3, sp_colors[sp_id], -1);

                std::string sp_text = std::to_string(sp_id);
                cv::putText(visualization, sp_text,
                    cv::Point(center.x + 5, center.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, sp_colors[sp_id], 1);
            }
        }

        return visualization;
    }

    AnalysisResult TextureAnalysis::analyzeTextureWithSuperpixelsStructured(
        const cv::Mat& input_image,
        const std::string& model_path,
        int region_size,
        float ruler) {

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

        std::vector<Edge> edges = extractEdgesStructured(input_image, edge_probability_map);
        if (edges.empty()) {
            std::cerr << "Any edge didn't found" << std::endl;
            return empty_result;
        }
        cv::Mat edges_visualization = visualizeEdges(input_image, edges);

        // Вычисляем суперпиксели
        cv::Mat superpixel_labels = computeSuperpixels(input_image, region_size, ruler);
        cv::Mat sp_visualization = visualizeSuperpixels(input_image, superpixel_labels);

        // Привязываем ребра к суперпикселям
        std::unordered_map<int, std::vector<Edge>> superpixel_edges;

        for (const auto& edge : edges) {
            cv::Point2f center = edge.getCenter();
            int x = static_cast<int>(center.x);
            int y = static_cast<int>(center.y);

            if (x >= 0 && x < superpixel_labels.cols && y >= 0 && y < superpixel_labels.rows) {
                int sp_id = superpixel_labels.at<int>(y, x);
                superpixel_edges[sp_id].push_back(edge);
            }
        }
        std::cout << "Superpixels with edges: " << superpixel_edges.size() << std::endl;

        std::vector<EdgeGroup> groups;
        std::vector<int> group_to_superpixel;

        for (const auto& [sp_id, sp_edges] : superpixel_edges) {
            std::cout << "  Superpixel " << sp_id << " has " << sp_edges.size() << " edges" << std::endl;

            EdgeGroup group(sp_edges);
            groups.push_back(group);
            group_to_superpixel.push_back(sp_id);
            std::cout << "    -> Created group " << groups.size() - 1 << std::endl;
        }

        //for (const auto& edge : edges) {
        //    cv::Point2f center = edge.getCenter();
        //    int x = static_cast<int>(center.x);
        //    int y = static_cast<int>(center.y);

        //    if (x >= 0 && x < superpixel_labels.cols && y >= 0 && y < superpixel_labels.rows) {
        //        int sp_id = superpixel_labels.at<int>(y, x);
        //        if (std::find())
        //        superpixel_edges[sp_id].push_back(edge);
        //    }
        //}

        std::cout << "Groups created: " << groups.size() << std::endl;
        std::cout << "group_to_superpixel size: " << group_to_superpixel.size() << std::endl;

        cv::Mat groups_visualization = visualizeGroups(input_image, groups);

        // Создаем композитную визуализацию
        cv::Mat composite = input_image.clone();
        cv::addWeighted(composite, 0.7, sp_visualization, 0.3, 0, composite);
        cv::addWeighted(composite, 1.0, edges_visualization, 0.5, 0, composite);

        cv::Mat prob_display;
        edge_probability_map.convertTo(prob_display, CV_8UC1, 255);

        // Создаем EBPT
        EBPT ebpt_model(input_image);
        for (const auto& group : groups) {
            ebpt_model.addEdgeGroup(group);
        }

        // Заполняем результат
        AnalysisResult result(ebpt_model, edges, groups,
            edges_visualization, groups_visualization,
            edge_probability_map);
        result.superpixel_visualization = sp_visualization;
        result.superpixel_labels = superpixel_labels;
        result.group_to_superpixel = group_to_superpixel;

        std::cout << "Final - groups: " << result.groups.size()
            << ", mapping: " << result.group_to_superpixel.size() << std::endl;

        return result;
    }

    cv::Mat TextureAnalysis::getSuperpixelMask(const cv::Mat& labels, int superpixel_id) {
        cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8UC1);

        for (int y = 0; y < labels.rows; ++y) {
            for (int x = 0; x < labels.cols; ++x) {
                if (labels.at<int>(y, x) == superpixel_id) {
                    mask.at<uchar>(y, x) = 255;
                }
            }
        }

        return mask;
    }
}