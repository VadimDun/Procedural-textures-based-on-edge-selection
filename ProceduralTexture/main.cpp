#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Edge.h"
#include "EdgeGroup.h"
#include "EBPT.h"
#include "TextureAnalysis.h"
#include "TextureSynthesis.h"
#include "PixelSynthesis.h"
#include "Config.h" // не придумал как спрятать путь от гита

using namespace EBPTns;

bool isCanny = false;

static cv::Mat createTestImage(int width = 400, int height = 400) {
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    image.setTo(cv::Scalar(100, 100, 100));

    cv::line(image, cv::Point(50, 50), cv::Point(150, 50), cv::Scalar(255, 0, 0), 3);
    cv::line(image, cv::Point(150, 50), cv::Point(100, 150), cv::Scalar(255, 0, 0), 3);
    cv::line(image, cv::Point(100, 150), cv::Point(50, 50), cv::Scalar(255, 0, 0), 3);

    cv::line(image, cv::Point(250, 100), cv::Point(350, 100), cv::Scalar(0, 255, 0), 3);
    cv::line(image, cv::Point(300, 50), cv::Point(300, 150), cv::Scalar(0, 255, 0), 3);

    cv::line(image, cv::Point(50, 250), cv::Point(150, 350), cv::Scalar(0, 0, 255), 3);

    cv::line(image, cv::Point(250, 300), cv::Point(350, 300), cv::Scalar(0, 255, 255), 3);

    return image;
}

static cv::Mat loadRealTexture(const std::string& path) {
    cv::Mat image = cv::imread(path);

    if (image.empty()) {
        return createTestImage(400, 400);
    }

    if (image.cols > 512 || image.rows > 512) {
        cv::resize(image, image, cv::Size(512, 512));
    }

    return image;
}

static void visualizeChainCode(const EBPTns::Edge& edge, cv::Mat& image,
    const cv::Scalar& color = cv::Scalar(0, 255, 0),
    bool show_angle = true) {
    const auto& points = edge.getPoints();
    const auto& chain = edge.getChainCode();

    if (points.size() < 2 || chain.empty()) return;

    // Рисуем стрелки цепного кода
    cv::Point prev = points[0];
    for (size_t i = 1; i < points.size(); ++i) {
        cv::line(image, prev, points[i], cv::Scalar(100, 100, 100), 1, cv::LINE_AA);

        cv::arrowedLine(image, prev, points[i],
            color,
            1,
            cv::LINE_AA,
            0,
            0.2);
        prev = points[i];
    }

    // Отмечаем начальную точку
    cv::circle(image, points[0], 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(image, "S",
        cv::Point(points[0].x + 5, points[0].y - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

    // Отмечаем конечную точку
    cv::circle(image, points.back(), 3, cv::Scalar(255, 0, 0), -1);
    cv::putText(image, "E",
        cv::Point(points.back().x + 5, points.back().y - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);

    // Визуализация направления угла (главной оси PCA)
    if (show_angle) {
        cv::Point2f center = edge.getCenter();
        float angle = edge.getAngle();

        float line_length = edge.getLength() * 0.8f;
        cv::Point2f direction(cos(angle) * line_length, sin(angle) * line_length);
        cv::Point start_point(static_cast<int>(center.x - direction.x * 0.5f),
            static_cast<int>(center.y - direction.y * 0.5f));
        cv::Point end_point(static_cast<int>(center.x + direction.x * 0.5f),
            static_cast<int>(center.y + direction.y * 0.5f));

        cv::arrowedLine(image, start_point, end_point,
            cv::Scalar(0, 255, 255), 2, cv::LINE_AA, 0, 0.3);

        std::string angle_text = std::to_string(static_cast<int>(angle * 180 / CV_PI)) + "°";
        cv::putText(image, angle_text,
            cv::Point(static_cast<int>(center.x + 10), static_cast<int>(center.y - 10)),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    }

    std::cout << "  Angle: " << static_cast<int>(edge.getAngle() * 180 / CV_PI) << "°" << std::endl;
}

static void visualizeAllChainCodes(const std::vector<EBPTns::Edge>& edges,
    cv::Mat& image,
    const std::string& filename = "images/chain_code_viz.png") {
    cv::Mat viz_image = image.clone();

    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255)
    };

    std::cout << "\n=== Chain code vizualization ===" << std::endl 
        << "Edges count: " << edges.size() << std::endl;

    //  тонкие серые линии
    //for (Edge edge : edges) {
    //    const auto& points = edge.getPoints();
    //    for (size_t i = 1; i < points.size(); ++i) {
    //        cv::line(viz_image, points[i - 1], points[i], cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
    //    }
    //}

    //цветные стрелки и направления
    for (size_t i = 0; i < edges.size(); ++i) {
        cv::Scalar color = colors[i % colors.size()];
        visualizeChainCode(edges[i], viz_image, color, true);
    }

    cv::imwrite(filename, viz_image);
}


static void visualizeAnglesOnly(const std::vector<EBPTns::Edge>& edges,
    const cv::Mat& background,
    const std::string& filename = "images/angles_only.png") {
    cv::Mat angle_viz = cv::Mat::zeros(background.size(), CV_8UC3);
    angle_viz.setTo(cv::Scalar(30, 30, 30));

    for (const auto& edge : edges) {
        cv::Point2f center = edge.getCenter();
        float angle = edge.getAngle();
        float length = edge.getLength() * 0.5f;

        cv::Point2f dir(cos(angle) * length, sin(angle) * length);
        cv::Point start(static_cast<int>(center.x - dir.x),
            static_cast<int>(center.y - dir.y));
        cv::Point end(static_cast<int>(center.x + dir.x),
            static_cast<int>(center.y + dir.y));

        int hue = static_cast<int>(angle * 180 / CV_PI) % 180;
        cv::Scalar color;
        if (hue < 60) color = cv::Scalar(0, 255, 0);       // Зеленый
        else if (hue < 120) color = cv::Scalar(255, 0, 0); // Синий
        else color = cv::Scalar(0, 0, 255);                // Красный

        cv::arrowedLine(angle_viz, start, end, color, 2, cv::LINE_AA, 0, 0.3);
        cv::circle(angle_viz, cv::Point(static_cast<int>(center.x),
            static_cast<int>(center.y)),
            3, cv::Scalar(255, 255, 255), -1);
    }

    cv::imwrite(filename, angle_viz);
}

void visualizeEdgeBins(const cv::Mat& input_image,
    const std::vector<EBPTns::Edge>& edges,
    const std::string& filename = "images/edge_bins.png") {

    if (edges.empty()) {
        std::cout << "No edges to visualize" << std::endl;
        return;
    }

    // Находим минимальную и максимальную длину для нормализации
    float min_length = edges[0].getLength();
    float max_length = edges[0].getLength();
    for (const auto& edge : edges) {
        float len = edge.getLength();
        if (len < min_length) min_length = len;
        if (len > max_length) max_length = len;
    }

    int img_width = input_image.cols;
    int img_height = input_image.rows;

    // Добавляем отступы между изображениями
    int padding = 20;
    int header_height = 40;

    // Создаем 9 изображений для бинов (каждое такого же размера, как исходное)
    std::vector<cv::Mat> bin_images(9);
    for (int i = 0; i < 9; ++i) {
        if (input_image.channels() == 1) {
            cv::cvtColor(input_image, bin_images[i], cv::COLOR_GRAY2BGR);
        }
        else {
            bin_images[i] = input_image.clone();
        }
        // Затемняем фон, чтобы ребра были видны ярче
        cv::addWeighted(bin_images[i], 0.3, cv::Mat::zeros(bin_images[i].size(), bin_images[i].type()), 0.7, 0, bin_images[i]);
    }

    // Распределяем ребра по бинам
    for (const auto& edge : edges) {
        float length = edge.getLength();
        float angle = edge.getAngle();  // 0 to PI

        // Нормализуем длину к [0, 2] (3 бина)
        float norm_length = 0.0f;
        if (max_length > min_length) {
            norm_length = (length - min_length) / (max_length - min_length) * 2.0f;
        }

        // Определяем бины
        int length_bin = static_cast<int>(norm_length);
        if (length_bin > 2) length_bin = 2;

        int angle_bin = 0;
        if (angle < CV_PI / 3) angle_bin = 0;           // 0-60°
        else if (angle < 2 * CV_PI / 3) angle_bin = 1;  // 60-120°
        else angle_bin = 2;                              // 120-180°

        int bin_index = length_bin * 3 + angle_bin;

        // Рисуем ребро ТОЛЬКО в соответствующем бине
        cv::Mat& bin_img = bin_images[bin_index];
        const auto& points = edge.getPoints();

        // Рисуем линии
        for (size_t i = 1; i < points.size(); ++i) {
            cv::line(bin_img, points[i - 1], points[i],
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        // Рисуем центр ребра
        cv::Point center(static_cast<int>(edge.getCenter().x),
            static_cast<int>(edge.getCenter().y));
        cv::circle(bin_img, center, 3, cv::Scalar(0, 0, 255), -1);
    }

    // Создаем итоговое изображение для отображения всех 9 бинов
    int grid_width = img_width * 3 + padding * 4;
    int grid_height = img_height * 3 + header_height + padding * 4;
    cv::Mat grid = cv::Mat::zeros(grid_height, grid_width, CV_8UC3);
    grid.setTo(cv::Scalar(50, 50, 50));

    // Заголовки
    std::vector<std::string> length_labels = { "Short", "Medium", "Long" };
    std::vector<std::string> angle_labels = { "0-60°", "60-120°", "120-180°" };

    // Добавляем заголовки для углов (сверху)
    for (int col = 0; col < 3; ++col) {
        int x = padding + col * (img_width + padding) + img_width / 2 - 30;
        cv::putText(grid, angle_labels[col],
            cv::Point(x, header_height / 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }

    // Добавляем заголовки для длины (слева)
    for (int row = 0; row < 3; ++row) {
        int y = header_height + padding + row * (img_height + padding) + img_height / 2;
        cv::putText(grid, length_labels[row],
            cv::Point(10, y),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }

    // Размещаем бины на сетке
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            int bin_index = row * 3 + col;

            int x = padding + col * (img_width + padding);
            int y = header_height + padding + row * (img_height + padding);

            // Проверяем, что ROI не выходит за границы
            if (x + img_width <= grid.cols && y + img_height <= grid.rows) {
                cv::Rect roi(x, y, img_width, img_height);
                bin_images[bin_index].copyTo(grid(roi));

                // Рамка вокруг бина
                cv::rectangle(grid, roi, cv::Scalar(200, 200, 200), 2);

                // Количество ребер в бине
                int edge_count = 0;
                for (const auto& edge : edges) {
                    float angle = edge.getAngle();
                    float length = edge.getLength();
                    float norm_length = (length - min_length) / (max_length - min_length) * 2.0f;
                    int l_bin = static_cast<int>(norm_length);
                    if (l_bin > 2) l_bin = 2;

                    int a_bin = 0;
                    if (angle < CV_PI / 3) a_bin = 0;
                    else if (angle < 2 * CV_PI / 3) a_bin = 1;
                    else a_bin = 2;

                    if (l_bin == row && a_bin == col) edge_count++;
                }

                std::string count_text = "n=" + std::to_string(edge_count);
                cv::putText(grid, count_text,
                    cv::Point(x + img_width - 80, y + img_height - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            }
        }
    }

    cv::imwrite(filename, grid);
    std::cout << "Bins visualization saved: " << filename << std::endl;
}

void visualizeBinDistribution(const std::vector<EBPTns::Edge>& edges,
    const std::string& filename = "images/bin_distribution.png") {

    if (edges.empty()) return;

    float min_len = edges[0].getLength();
    float max_len = edges[0].getLength();
    for (const auto& e : edges) {
        min_len = std::min(min_len, e.getLength());
        max_len = std::max(max_len, e.getLength());
    }

    int bins[3][3] = { {0} };

    for (const auto& edge : edges) {
        float len = edge.getLength();
        float angle = edge.getAngle();

        int len_bin;
        if (max_len - min_len < 0.1f) len_bin = 1;
        else {
            float norm = (len - min_len) / (max_len - min_len);
            if (norm < 0.33f) len_bin = 0;
            else if (norm < 0.66f) len_bin = 1;
            else len_bin = 2;
        }

        int angle_bin;
        if (angle < CV_PI / 3) angle_bin = 0;
        else if (angle < 2 * CV_PI / 3) angle_bin = 1;
        else angle_bin = 2;

        bins[len_bin][angle_bin]++;
    }

    int cell_size = 100;
    cv::Mat dist_viz = cv::Mat::zeros(cell_size * 4, cell_size * 4, CV_8UC3);
    dist_viz.setTo(cv::Scalar(240, 240, 240));

    std::vector<std::string> len_labels = { "Short", "Medium", "Long" };
    std::vector<std::string> angle_labels = { "0-60°", "60-120°", "120-180°" };

    // Рисуем таблицу
    for (int i = 0; i <= 3; ++i) {
        cv::line(dist_viz, cv::Point(i * cell_size, 0),
            cv::Point(i * cell_size, cell_size * 3),
            cv::Scalar(0, 0, 0), 2);
        cv::line(dist_viz, cv::Point(0, i * cell_size),
            cv::Point(cell_size * 3, i * cell_size),
            cv::Scalar(0, 0, 0), 2);
    }

    // Заполняем ячейки
    int total = edges.size();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            int count = bins[row][col];
            float percent = 100.0f * count / total;

            cv::Rect cell(col * cell_size, row * cell_size, cell_size, cell_size);

            int intensity = static_cast<int>(120 + 100 * percent / 50);
            intensity = std::min(255, intensity);
            cv::rectangle(dist_viz, cell, cv::Scalar(255 - intensity, 255, 255 - intensity), -1);

            std::string text = std::to_string(count);
            cv::putText(dist_viz, text,
                cv::Point(col * cell_size + 20, row * cell_size + 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

            std::string percent_text = std::to_string(static_cast<int>(percent)) + "%";
            cv::putText(dist_viz, percent_text,
                cv::Point(col * cell_size + 20, row * cell_size + 70),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 80, 80), 1);
        }
    }

    for (int i = 0; i < 3; ++i) {
        cv::putText(dist_viz, angle_labels[i],
            cv::Point(i * cell_size + 20, cell_size * 3 + 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        cv::putText(dist_viz, len_labels[i],
            cv::Point(cell_size * 3 + 10, i * cell_size + 40),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    cv::putText(dist_viz, "Angle →", cv::Point(cell_size, cell_size * 3 + 50),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::putText(dist_viz, "Length ↓", cv::Point(cell_size * 3 + 20, cell_size),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    cv::imwrite(filename, dist_viz);
    std::cout << "Распределение по бинам: " << filename << std::endl;

    std::cout << "\nТаблица распределения ребер по бинам (длина × угол):" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "          0-60°   60-120°  120-180°  Total" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    for (int row = 0; row < 3; ++row) {
        std::cout << len_labels[row] << "   ";
        int row_total = 0;
        for (int col = 0; col < 3; ++col) {
            std::cout << "  " << bins[row][col] << "\t";
            row_total += bins[row][col];
        }
        std::cout << "  " << row_total << std::endl;
    }
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Total: " << total << " ребер" << std::endl;
}


int main(int argc, char** argv) {
    setlocale(LC_ALL, "ru");

    bool use_real_texture = false;
    std::string texture_path;

    if (argc > 1) {
        texture_path = argv[1];
        use_real_texture = true;
    }

    // Загружаем изображение
    cv::Mat input_image;

    if (use_real_texture) {
        input_image = loadRealTexture(texture_path);
    }
    else {
        input_image = createTestImage(400, 400);
    }

    if (input_image.empty()) {
        return -1;
    }

    cv::imwrite("images/input_texture.png", input_image);

    // Анализ текстуры
    TextureAnalysis analyzer;

    if (use_real_texture) {
        analyzer.setCannyThresholds(50, 150);
        analyzer.setMinEdgeLength(15);
        analyzer.setGroupingDistance(20);
    }
    else {
        analyzer.setCannyThresholds(20, 60);
        analyzer.setMinEdgeLength(10);
        analyzer.setGroupingDistance(60);
    }

    //std::vector<Edge> edges = analyzer.extractEdges(input_image);
    //cv::Mat edges_visualization = analyzer.visualizeEdges(input_image, edges);
    //cv::imwrite("images/edges_detected.png", edges_visualization);

    //std::vector<EdgeGroup> groups = analyzer.groupEdges(edges);
    //cv::Mat groups_visualization = analyzer.visualizeGroups(input_image, groups);
    //cv::imwrite("images/groups_detected.png", groups_visualization);


    //// Создаем EBPT модель
    //EBPT ebpt_model(input_image);
    //for (const auto& group : groups) {
    //    ebpt_model.addEdgeGroup(group);
    //}
    //EBPT ebpt_model = analyzer.analyzeTexture(input_image);
    //EBPT ebpt_model = analyzer.analyzeTextureStructured(input_image, MODEL_PATH);

    AnalysisResult result;
    cv::Mat prob_map;

    if (isCanny) {
        // Canny
        result = analyzer.analyzeTexture(input_image);
    }
    else {
        // Structured forests
        result = analyzer.analyzeTextureStructured(input_image, MODEL_PATH);
        prob_map = result.edge_probability_map;
    }

    EBPT ebpt_model = result.modelEBPT;
    std::vector<Edge> edges = result.edges;
    std::vector<EdgeGroup> groups = result.groups;
    cv::Mat edges_visualization = result.edges_visualization;
    cv::Mat groups_visualization = result.groups_visualization;

    if (!result.isValid()) { return 1; }

    //////////////////////////////////
    // Цепной код и PCA
    /////////////////////////////////

    cv::Mat chain_viz = input_image.clone();
    visualizeAllChainCodes(edges, chain_viz, "images/chain_code_debug.png");
    visualizeAnglesOnly(result.edges, input_image, "images/angles_directions.png");

    /////////////////////////////
    // Карта вероятностей
    /////////////////////////////

    cv::Mat prob_map_display = result.edge_probability_map * 255;
    prob_map_display.convertTo(prob_map_display, CV_8UC1);
    cv::imwrite("images/probability_map.png", prob_map_display);

    //double thresholds[] = { 0.2, 0.4, 0.6, 0.8 };
    //for (double th : thresholds) {
    //    cv::Mat binary;
    //    cv::threshold(result.edge_probability_map, binary, th, 255, cv::THRESH_BINARY);
    //    binary.convertTo(binary, CV_8UC1);
    //    cv::imwrite("images/edges_th_" + std::to_string(th) + ".png", binary);
    //}

    //int histSize = 20;
    //float range[] = { 0, 1 };
    //const float* histRange = { range };
    //cv::Mat hist;
    //cv::calcHist(&result.edge_probability_map, 1, 0, cv::Mat(),
    //    hist, 1, &histSize, &histRange);

     //Нормализуем для отображения
    //cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

    // Смотрим результат
    //std::cout << "Распределение вероятностей:" << std::endl;
    //for (int i = 0; i < histSize; i++) {
    //    float bin_start = i * (1.0f / histSize);
    //    float bin_end = (i + 1) * (1.0f / histSize);
    //    float value = hist.at<float>(i);
    //    std::cout << "  " << bin_start << "-" << bin_end << ": "
    //        << value << " пикселей" << std::endl;
    //}

    /////////////////////////////
    // Бины
    /////////////////////////////

    visualizeEdgeBins(input_image, result.edges, "images/edge_bins_structured.png");

    visualizeBinDistribution(result.edges, "images/bin_distribution.png");

    /////////////////////////////
    // Суперпиксели
    /////////////////////////////
    auto result1 = analyzer.analyzeTextureWithSuperpixelsStructured(input_image, MODEL_PATH, 120, 10.0f);
    cv::imwrite("images/superpixels_boundaries.png", result1.superpixel_visualization);

    // Создаем композитное изображение: исходное + суперпиксели + ребра
    cv::Mat composite = input_image.clone();
    cv::addWeighted(composite, 0.7, result1.superpixel_visualization, 0.3, 0, composite);
    cv::addWeighted(composite, 1.0, result1.edges_visualization, 0.5, 0, composite);
    cv::imwrite("images/superpixels_with_edges.png", composite);

    //////////////////////////////////
    float scale = 1.0f;
    float density = 0.7f;
    float angle_spread = 0.3f;

    if (use_real_texture) {
        scale = 0.8f;
        density = 0.6f;
        angle_spread = 0.4f;
    }

    ebpt_model.setScale(scale);
    ebpt_model.setDensity(density);
    ebpt_model.setAngleSpread(angle_spread);

    // Синтез размещения
    TextureSynthesis synthesizer;
    synthesizer.setRandomSeed(42);
    synthesizer.setAvoidOverlap(true);
    synthesizer.setMinDistance(40.0f);

    const auto& source_groups = ebpt_model.getEdgeGroups();

    int output_width, output_height;

    if (use_real_texture) {
        output_width = input_image.cols * 2;
        output_height = input_image.rows * 2;
    }
    else {
        output_width = 800;
        output_height = 600;
    }

    float synth_density = density * 1.5f;
    float synth_angle_variation = angle_spread * 1.2f;
    float synth_scale_variation = 0.25f;

    std::vector<PlacedGroup> placed_groups = synthesizer.synthesizePlacement(
        source_groups,
        output_width, output_height,
        synth_density,
        synth_angle_variation,
        synth_scale_variation
    );

    // Визуализируем размещение
    cv::Mat placement_map = synthesizer.drawPlacementMap(
        placed_groups, output_width, output_height
    );
    cv::imwrite("images/placement_map.png", placement_map);

    // Заполнение пикселей
    PixelSynthesis pixel_synthesis;
    pixel_synthesis.setRandomSeed(123);
    pixel_synthesis.setPatchSelectionMethod(1);

    cv::Mat output_texture = pixel_synthesis.fillPixels(
        input_image,
        groups,
        placed_groups,
        output_width, output_height
    );

    cv::imwrite("images/output_texture.png", output_texture);

    // Создаем итоговую визуализацию
    int viz_width = 1200;
    int viz_height = 800;
    cv::Mat final_visualization = cv::Mat::zeros(viz_height, viz_width, CV_8UC3);
    final_visualization.setTo(cv::Scalar(30, 30, 30));

    cv::Mat input_resized;
    cv::resize(input_image, input_resized, cv::Size(350, 350));
    input_resized.copyTo(final_visualization(cv::Rect(50, 50, 350, 350)));
    cv::rectangle(final_visualization, cv::Rect(50, 50, 350, 350),
        cv::Scalar(255, 255, 255), 2);
    cv::putText(final_visualization, "Input Texture",
        cv::Point(60, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2);

    cv::Mat edges_resized;
    cv::resize(edges_visualization, edges_resized, cv::Size(350, 350));
    edges_resized.copyTo(final_visualization(cv::Rect(450, 50, 350, 350)));
    cv::rectangle(final_visualization, cv::Rect(450, 50, 350, 350),
        cv::Scalar(255, 255, 255), 2);
    cv::putText(final_visualization, "Detected Edges (" + std::to_string(edges.size()) + ")",
        cv::Point(460, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2);

    cv::Mat groups_resized;
    cv::resize(groups_visualization, groups_resized, cv::Size(350, 350));
    groups_resized.copyTo(final_visualization(cv::Rect(50, 450, 350, 350)));
    cv::rectangle(final_visualization, cv::Rect(50, 450, 350, 350),
        cv::Scalar(255, 255, 255), 2);
    cv::putText(final_visualization, "Edge Groups (" + std::to_string(groups.size()) + ")",
        cv::Point(60, 440), cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2);

    cv::Mat output_resized;
    cv::resize(output_texture, output_resized, cv::Size(350, 350));
    output_resized.copyTo(final_visualization(cv::Rect(450, 450, 350, 350)));
    cv::rectangle(final_visualization, cv::Rect(450, 450, 350, 350),
        cv::Scalar(255, 255, 255), 2);
    cv::putText(final_visualization, "Output Texture",
        cv::Point(460, 440), cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2);

    cv::Mat map_resized;
    cv::resize(placement_map, map_resized, cv::Size(350, 350));
    map_resized.copyTo(final_visualization(cv::Rect(825, 50, 350, 350)));
    cv::rectangle(final_visualization, cv::Rect(825, 50, 350, 350),
        cv::Scalar(255, 255, 255), 2);
    cv::putText(final_visualization, "Placement Map",
        cv::Point(835, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2);

    cv::Rect info_rect(825, 450, 350, 350);
    cv::rectangle(final_visualization, info_rect, cv::Scalar(50, 50, 70), -1);
    cv::rectangle(final_visualization, info_rect, cv::Scalar(200, 200, 200), 2);

    std::vector<std::string> info_lines = {
        "EBPT Texture Synthesis Report",
        "=============================",
        "Configuration:",
        "  Input: " + std::to_string(input_image.cols) + "x" +
                    std::to_string(input_image.rows),
        "  Output: " + std::to_string(output_width) + "x" +
                     std::to_string(output_height),
        "",
        "Analysis Results:",
        "  Edges detected: " + std::to_string(edges.size()),
        "  Groups created: " + std::to_string(groups.size()),
        "",
        "Synthesis Parameters:",
        "  Scale: " + std::to_string(scale),
        "  Density: " + std::to_string(density),
        "  Angle spread: " + std::to_string(angle_spread),
        "",
        "Placement Results:",
        "  Groups placed: " + std::to_string(placed_groups.size()),
        "  Texture type: " + std::string(use_real_texture ? "Real" : "Test")
    };

    for (size_t i = 0; i < info_lines.size(); ++i) {
        cv::putText(final_visualization, info_lines[i],
            cv::Point(info_rect.x + 10, info_rect.y + 30 + i * 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.45,
            cv::Scalar(200, 200, 150), 1);
    }

    cv::putText(final_visualization, "System Info",
        cv::Point(info_rect.x + 10, info_rect.y + 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.6,
        cv::Scalar(255, 255, 255), 1);

    cv::imwrite("images/final_report.png", final_visualization);

    cv::namedWindow("Input Texture", cv::WINDOW_NORMAL);
    cv::imshow("Input Texture", input_image);

    cv::namedWindow("Detected Edges", cv::WINDOW_NORMAL);
    cv::imshow("Detected Edges", edges_visualization);

    cv::namedWindow("Edge Groups", cv::WINDOW_NORMAL);
    cv::imshow("Edge Groups", groups_visualization);

    cv::namedWindow("Placement Map", cv::WINDOW_NORMAL);
    cv::imshow("Placement Map", placement_map);

    cv::namedWindow("Output Texture", cv::WINDOW_NORMAL);
    cv::imshow("Output Texture", output_texture);

    //cv::namedWindow("Final Report", cv::WINDOW_NORMAL);
    //cv::imshow("Final Report", final_visualization);

    bool show_edges = true;
    bool show_groups = true;
    bool running = true;

    while (running) {
        int key = cv::waitKey(30);

        switch (key) {
        case 'q':
        case 'Q':
        case 27:
            running = false;
            break;

        case 's':
        case 'S':
            cv::imwrite("images/current_input.png", input_image);
            cv::imwrite("images/current_edges.png", edges_visualization);
            cv::imwrite("images/current_groups.png", groups_visualization);
            cv::imwrite("images/current_placement.png", placement_map);
            cv::imwrite("images/current_output.png", output_texture);
            break;

        case 'к':
        case 'К':
        case 'r':
        case 'R':
            placed_groups = synthesizer.synthesizePlacement(
                source_groups, output_width, output_height,
                synth_density, synth_angle_variation, synth_scale_variation
            );
            placement_map = synthesizer.drawPlacementMap(
                placed_groups, output_width, output_height
            );
            output_texture = pixel_synthesis.fillPixels(
                input_image, groups, placed_groups, output_width, output_height
            );
            cv::imshow("Placement Map", placement_map);
            cv::imshow("Output Texture", output_texture);
            break;

        case 'n':
        case 'N':
        {
            unsigned int new_seed = std::random_device{}();
            synthesizer.setRandomSeed(new_seed);
            pixel_synthesis.setRandomSeed(new_seed);
        }
        break;

        case '1':
            show_edges = !show_edges;
            if (show_edges) {
                cv::imshow("Detected Edges", edges_visualization);
            }
            else {
                cv::destroyWindow("Detected Edges");
            }
            break;

        case '2':
            show_groups = !show_groups;
            if (show_groups) {
                cv::imshow("Edge Groups", groups_visualization);
            }
            else {
                cv::destroyWindow("Edge Groups");
            }
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}