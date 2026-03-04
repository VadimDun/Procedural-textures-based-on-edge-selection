#include "TextureAnalysis.h"
#include "ImageDisplay.h"
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

    //AnalysisResult TextureAnalysis::analyzeTextureStructured(
    //    const cv::Mat& input_image,
    //    const std::string& model_path) {

    //    std::cout << "Picture size: " << input_image.cols << "x" << input_image.rows << std::endl;

    //    AnalysisResult empty_result;

    //    if (!initializeStructuredDetector(model_path)) {
    //        std::cerr << "TextureAnalysis::analyzeTextureStructured: StructuredForests didn't created" << std::endl;
    //        return empty_result;
    //    }

    //    // Конвертируем в float и нормализуем
    //    cv::Mat float_image;
    //    if (input_image.channels() == 3) {
    //        cv::cvtColor(input_image, float_image, cv::COLOR_BGR2RGB); // Модель обучена на RGB
    //    }
    //    else {
    //        cv::cvtColor(input_image, float_image, cv::COLOR_GRAY2RGB); // Если grayscale, делаем 3 канала
    //    }
    //    float_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    //    cv::Mat edge_probability_map;
    //    structured_edge_detector_->detectEdges(float_image, edge_probability_map);

    //    // Получаем рёбра
    //    std::vector<Edge> edges = extractEdgesStructured(input_image, edge_probability_map);
    //    if (edges.empty()) {
    //        std::cerr << "Any edge didn't found" << std::endl;
    //        return empty_result;
    //    }

    //    std::vector<EdgeGroup> groups = groupEdges(edges);
    //    if (groups.empty()) {
    //        std::cerr << "Any group didn't created" << std::endl;
    //        return empty_result;
    //    }

    //    // Создаем EBPT
    //    EBPT ebpt_model(input_image);
    //    for (const auto& group : groups) {
    //        ebpt_model.addEdgeGroup(group);
    //    }

    //    // Создаем визуализации для отладки. Todo потом может удалить
    //    cv::Mat edges_visualization = visualizeEdges(input_image, edges);
    //    cv::Mat groups_visualization = visualizeGroups(input_image, groups);

    //    std::cout << "   Edges found: " << edges.size() << std::endl;
    //    std::cout << "   Groups created: " << groups.size() << std::endl;

    //    return AnalysisResult(ebpt_model, edges, groups,
    //        edges_visualization, groups_visualization,
    //        edge_probability_map);
    //}

    //AnalysisResult TextureAnalysis::analyzeTexture(const cv::Mat& input_image) {
    //    AnalysisResult result;

    //    result.edges = extractEdges(input_image);
    //    result.edges_visualization = visualizeEdges(input_image, result.edges);

    //    result.groups = groupEdges(result.edges);
    //    result.groups_visualization = visualizeGroups(input_image, result.groups);

    //    result.modelEBPT = EBPT(input_image);
    //    for (const auto& group : result.groups) {
    //        result.modelEBPT.addEdgeGroup(group);
    //    }

    //    return result;
    //}

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
        cv::Mat edges_visualization = ImageDisplay::visualizeEdges(input_image, edges);
        ImageDisplay::setPartFinalVisualization(edges_visualization, ImageDisplay::edges);

        // Вычисляем суперпиксели
        cv::Mat superpixel_labels = computeSuperpixels(input_image, region_size, ruler);
        cv::Mat sp_visualization = ImageDisplay::visualizeSuperpixels(input_image, superpixel_labels);
        ImageDisplay::saveAndShow("superpixels_boundaries.png", "Superpixels", sp_visualization);

        std::vector<SourceGroupInfo> source_infos;
        for (const auto& edge : edges) {
            cv::Point2f center = edge.getCenter();
            int x = static_cast<int>(center.x);
            int y = static_cast<int>(center.y);

            if (x >= 0 && x < superpixel_labels.cols && y >= 0 && y < superpixel_labels.rows) {
                int sp_id = superpixel_labels.at<int>(y, x);

                if (std::count_if(source_infos.begin(), source_infos.end(),
                    [sp_id](const SourceGroupInfo& sg) { return sg.superpixel_id == sp_id; }) == 0) {
                    SourceGroupInfo info;
                    info.group = EdgeGroup(edge);
                    info.superpixel_id = sp_id;
                    info.superpixel_mask = getSuperpixelMask(superpixel_labels, sp_id);
                    source_infos.push_back(info);
                }
                else {
                    auto it = std::find_if(source_infos.begin(), source_infos.end(),
                        [sp_id](const SourceGroupInfo& sg) { return sg.superpixel_id == sp_id; });

                    if (it != source_infos.end()) {
                        SourceGroupInfo& info = *it;
                        info.group.addEdge(edge);
                    }
                }

            }
        }

        std::cout << "Groups created: " << source_infos.size() << std::endl;

        cv::Mat groups_visualization = ImageDisplay::visualizeGroups(input_image, source_infos);
        ImageDisplay::setPartFinalVisualization(groups_visualization, ImageDisplay::groups);

        // Создаем композитное изображение: исходное + суперпиксели + ребра
        //ImageDisplay::visualiseSPWithEdges(input_image, sp_visualization, edges_visualization);

        cv::Mat prob_display;
        edge_probability_map.convertTo(prob_display, CV_8UC1, 255);

        // Создаем EBPT
        EBPT ebpt_model(input_image);
        for (const auto& group : source_infos) {
            ebpt_model.addEdgeGroup(group);
        }

        // Заполняем результат
        AnalysisResult result(ebpt_model, edges,
            edges_visualization, groups_visualization,
            edge_probability_map);
        result.superpixel_labels = superpixel_labels;

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