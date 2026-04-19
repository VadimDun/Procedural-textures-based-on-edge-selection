#include "TextureAnalysis.h"
#include "ImageDisplay.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <numeric>

namespace EBPTns {

    TextureAnalysis::TextureAnalysis() {
        setMinEdgeLength(10);
    }

    void TextureAnalysis::setMinEdgeLength(int min_length) {
        min_edge_length_ = min_length;
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
        //double threshold = 0.25; // todo сделать изменение порога в ручном формате в процессе работы программы, либо автоматический подбор
        cv::threshold(edge_probability_map, binary_edges, superpixel_threshold, 255, cv::THRESH_BINARY);
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

    /////////////////////////////
    // Суперпиксели
    /////////////////////////////
    void TextureAnalysis::setSuperpixelParams(int region_size, float ruler, double sp_threshold) {
        superpixel_region_size_ = region_size;
        superpixel_ruler_ = ruler;
        superpixel_threshold = sp_threshold;
        std::cout << "Параметры суперпикселей: region_size=" << region_size
            << ", ruler=" << ruler << ", threshold=" << sp_threshold << std::endl;
    }

    cv::Mat TextureAnalysis::computeSuperpixels(const cv::Mat& image) const {

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
                superpixel_region_size_,
                superpixel_ruler_);

        slic->iterate(5);

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

    void TextureAnalysis::classifySourceGroups(std::vector<SourceGroupInfo>& source_groups) {
        const bool USE_RADIAL_SPRED = true;
        if (source_groups.empty()) return;

        std::vector<float> group_sizes;
        group_sizes.reserve(source_groups.size());
        std::cout << "source_groups.size()=" << source_groups.size() << std::endl;

        for (const auto& group_info : source_groups) {
            float size = USE_RADIAL_SPRED ? group_info.group.getRadialSpread() : cv::contourArea(group_info.hull);
            std::cout << "group_sizes " << size << std::endl;
            if (size == 0) {
                continue;

            }
            group_sizes.push_back(size);
        }

        std::vector<float> sorted_sizes = group_sizes;
        std::sort(sorted_sizes.begin(), sorted_sizes.end());

        size_t n = sorted_sizes.size();
        std::cout << "n=" << n << std::endl;

        // СТРАТЕГИЯ 1: Поиск естественных разрывов
        std::vector<float> gaps;

        // Находим наибольшие разрывы между последовательными размерами
        for (size_t i = 1; i < n; ++i) {
            float gap = sorted_sizes[i] - sorted_sizes[i - 1];
            gaps.push_back(gap);
        }

        // Сортируем разрывы по убыванию
        std::vector<int> sorted_indices(gaps.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&gaps](int a, int b) { return gaps[a] > gaps[b]; });

        for (size_t i = 0; i < n-1; ++i) {
            std::cout << "sorted gap " << i << ": " << gaps[sorted_indices[i]] << std::endl;
        }

        // Используем 2 наибольших разрыва для разделения на 3 категории (крупные/средние/мелкие)
        if (n >= 4 && sorted_indices.size() >= 2) {
            int gap1_idx = sorted_indices[0];
            int split1 = gap1_idx + 1;

            int gap2_idx = sorted_indices[1];
            int split2 = gap2_idx + 1;

            // взяли два самых больших разрыва. разрыв с меньшим индексом разделяет large и 
            // medium(так как меньший индекс ближе к началу в отсортированном по размеру массиве)
            int lower_split = std::min(split1, split2);
            int upper_split = std::max(split1, split2);

            // Устанавливаем пороги на основе разрывов
            large_scale_threshold_ = sorted_sizes[upper_split];
            medium_scale_threshold_ = sorted_sizes[lower_split];

            std::cout << "Using natural breakpoints strategy:" << std::endl;
        }
        // СТРАТЕГИЯ 2: На основе среднего и стандартного отклонения
        else if (n >= 3) {
            float mean = 0.0f;
            for (float s : group_sizes) mean += s;
            mean /= n;

            float variance = 0.0f;
            for (float s : group_sizes) {
                variance += (s - mean) * (s - mean);
            }
            variance /= n;
            float std_dev = std::sqrt(variance);

            // Пороги: mean ± 0.5*std_dev (можно настроить коэффициент)
            large_scale_threshold_ = mean + 0.5f * std_dev;
            medium_scale_threshold_ = mean - 0.5f * std_dev;

            std::cout << "Using statistical thresholds (mean=" << mean
                << ", std=" << std_dev << "):" << std::endl;
        }
        else {
            float min_size = sorted_sizes[0];
            float max_size = sorted_sizes[n - 1];
            float range = max_size - min_size;

            large_scale_threshold_ = min_size + range * 0.75f;
            medium_scale_threshold_ = min_size + range * 0.5f;
            std::cout << "Using area:" << std::endl;
        }

        std::cout << "Scale thresholds: LARGE >= " << large_scale_threshold_
            << ", MEDIUM >= " << medium_scale_threshold_
            << ", SMALL < " << medium_scale_threshold_ << std::endl;

        // Классифицируем каждую группу
        for (auto& group_info : source_groups) {
            float size = USE_RADIAL_SPRED ? group_info.group.getRadialSpread() : cv::contourArea(group_info.hull);

            if (size >= large_scale_threshold_) {
                group_info.scale_level = ScaleLevel::LARGE;
            }
            else if (size >= medium_scale_threshold_) {
                group_info.scale_level = ScaleLevel::MEDIUM;
            }
            else {
                group_info.scale_level = ScaleLevel::SMALL;
            }

            //std::cout << "Group " << (&group_info - &source_groups[0])
            //    << ": size=" << size
            //    << ", level=" << scaleLevelToString(group_info.scale_level) << std::endl;
        }

        checkAndAdjustThresholds(source_groups);
    }

    // TODO сделать проверку каждой группы
    // Вспомогательный метод для проверки и коррекции
    void TextureAnalysis::checkAndAdjustThresholds(std::vector<SourceGroupInfo>& source_groups) {
        int large_count = 0, medium_count = 0, small_count = 0;

        for (const auto& group : source_groups) {
            switch (group.scale_level) {
            case ScaleLevel::LARGE: large_count++; break;
            case ScaleLevel::MEDIUM: medium_count++; break;
            case ScaleLevel::SMALL: small_count++; break;
            }
        }

        // Если более 80% групп в одном уровне, корректируем пороги
        int total = source_groups.size();
        if (large_count > total * 0.8f) {
            std::cout << "Warning: Too many LARGE groups, adjusting thresholds..." << std::endl;
            large_scale_threshold_ *= 0.7f;
            medium_scale_threshold_ *= 0.7f;
            // Переклассифицируем
            for (auto& group : source_groups) {
                float size = group.group.getRadialSpread();
                if (size >= large_scale_threshold_) {
                    group.scale_level = ScaleLevel::LARGE;
                }
                else if (size >= medium_scale_threshold_) {
                    group.scale_level = ScaleLevel::MEDIUM;
                }
                else {
                    group.scale_level = ScaleLevel::SMALL;
                }
            }
        }
    }

    AnalysisResult TextureAnalysis::analyzeTextureWithSuperpixelsStructured(
        const cv::Mat& input_image,
        const std::string& model_path) {

        std::cout << "Picture size: " << input_image.cols << "x" << input_image.rows << std::endl;

        if (!initializeStructuredDetector(model_path)) {
            std::cerr << "TextureAnalysis::analyzeTextureStructured: StructuredForests didn't created" << std::endl;
            return AnalysisResult();
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

        // Probability_map
        cv::Mat edge_probability_map;
        structured_edge_detector_->detectEdges(float_image, edge_probability_map);

        cv::Mat prob_map_display = edge_probability_map;
        prob_map_display.convertTo(prob_map_display, CV_8UC1, 255);
        //ImageDisplay::saveAndShow("probability_map.png", "probability", prob_map_display);

        // Edges
        std::vector<Edge> edges = extractEdgesStructured(input_image, edge_probability_map);
        if (edges.empty()) {
            std::cerr << "Any edge didn't found" << std::endl;
            return AnalysisResult();
        }
        //cv::Mat edges_visualization = ImageDisplay::visualizeEdges(input_image, edges);
        //ImageDisplay::setPartFinalVisualization(edges_visualization, ImageDisplay::edges);

        ImageDisplay::visualizeAllChainCodes(edges, input_image, "images/chain_code_debug.png");
        //ImageDisplay::visualizeAnglesOnly(edges, input_image, "images/angles_directions.png");

        // Вычисляем суперпиксели
        cv::Mat superpixel_labels = computeSuperpixels(input_image);
        //cv::Mat sp_visualization = ImageDisplay::visualizeSuperpixels(input_image, superpixel_labels);
        //ImageDisplay::saveAndShow("superpixels_boundaries.png", "Superpixels", sp_visualization);

        // Groups
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

        // Создаем композитное изображение: исходное + суперпиксели + ребра
        //ImageDisplay::visualiseSPWithEdges(input_image, sp_visualization, edges_visualization);

        std::sort(source_infos.begin(), source_infos.end(),
        [](const SourceGroupInfo& a, const SourceGroupInfo& b) {
            return a.group.getRadialSpread() > b.group.getRadialSpread();
        });
        classifySourceGroups(source_infos);

        EBPT ebpt_model;
        cv::Size size = input_image.size();

        int cnt = 1;
        for (auto& group : source_infos) {
            group.hull = computeHull(group.group);
            group.group.setIndex(cnt++);
            ebpt_model.addEdgeGroup(group);
            std::string s = "Patch before" + std::to_string(cnt);
            std::string s1 = "Mask before" + std::to_string(cnt);
            //ImageDisplay::show(s1, group.mask);
        }


        cv::Mat groups_hull_visualization = ImageDisplay::visualizeGroups(input_image, source_infos);
        ImageDisplay::saveAndShowWithSize("groups.png", "Edge Groups", groups_hull_visualization, cv::Size(groups_hull_visualization.cols, groups_hull_visualization.rows));

        //ImageDisplay::setPartFinalVisualization(groups_hull_visualization, ImageDisplay::groups);

        AnalysisResult result(ebpt_model, superpixel_labels);

        return result;
    }

    std::vector<cv::Point> TextureAnalysis::computeHull(const EdgeGroup& group) {
        std::vector<cv::Point> all_points = group.getAllPoints();
        std::vector<cv::Point> hull;

        if (all_points.empty()) {
            return hull;
        }

        cv::convexHull(all_points, hull);
        return hull;
    }
}