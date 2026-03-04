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
#include "ImageDisplay.h"

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


int main(int argc, char** argv) {
    setlocale(LC_ALL, "ru");

    ImageDisplay::initFinalVisualization();

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

    ImageDisplay::setPartFinalVisualization(input_image, ImageDisplay::input );
    //cv::imwrite("images/input_texture.png", input_image);

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

    cv::Mat prob_map;
    //AnalysisResult result;

    //if (isCanny) {
    //    // Canny
    //    result = analyzer.analyzeTexture(input_image);
    //}
    //else {
    //    // Structured forests
    //    result = analyzer.analyzeTextureStructured(input_image, MODEL_PATH);
    //    prob_map = result.edge_probability_map;
    //}

    /////////////////////////////
// Суперпиксели
/////////////////////////////
    auto result = analyzer.analyzeTextureWithSuperpixelsStructured(input_image, MODEL_PATH, 120, 10.0f);


    EBPT ebpt_model = result.modelEBPT;
    std::vector<Edge> edges = result.edges;
    //std::vector<EdgeGroup> groups = result.modelEBPT.groups;

    if (!result.isValid()) { return 1; }

    //////////////////////////////////
    // Цепной код и PCA
    /////////////////////////////////

    cv::Mat chain_viz = input_image.clone();
    ImageDisplay::visualizeAllChainCodes(edges, chain_viz, "images/chain_code_debug.png");
    ImageDisplay::visualizeAnglesOnly(result.edges, input_image, "images/angles_directions.png");

    /////////////////////////////
    // Карта вероятностей
    /////////////////////////////

    cv::Mat prob_map_display = result.edge_probability_map * 255;
    prob_map_display.convertTo(prob_map_display, CV_8UC1);
    ImageDisplay::save("images/probability_map.png", prob_map_display);

    //double thresholds[] = { 0.2, 0.4, 0.6, 0.8 };
    //for (double th : thresholds) {
    //    cv::Mat binary;
    //    cv::threshold(result.edge_probability_map, binary, th, 255, cv::THRESH_BINARY);
    //    binary.convertTo(binary, CV_8UC1);
    //    ImageDisplay::save("images/edges_th_" + std::to_string(th) + ".png", binary);
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

    //ImageDisplay::visualizeEdgeBins(input_image, result.edges, "images/edge_bins_structured.png");

    //ImageDisplay::visualizeBinDistribution(result.edges, "images/bin_distribution.png");

    //////////////////////////////////


    //std::cout << "Groups size: " << result.groups.size() << std::endl;
    //std::cout << "group_to_superpixel size: " << result.group_to_superpixel.size() << std::endl;

    //// Проверяем, что векторы не пусты и имеют одинаковый размер
    //if (result.groups.empty() || result.group_to_superpixel.empty()) {
    //    std::cerr << "Error: No groups or superpixel mapping found" << std::endl;
    //    return -1;
    //}

    //if (result.groups.size() != result.group_to_superpixel.size()) {
    //    std::cerr << "Error: Mismatch between groups and superpixel mapping" << std::endl;
    //    std::cerr << "Groups: " << result.groups.size()
    //        << ", Mapping: " << result.group_to_superpixel.size() << std::endl;
    //    return -1;
    //}

    //// Проверяем superpixel_labels
    //if (result.superpixel_labels.empty()) {
    //    std::cerr << "Error: superpixel_labels is empty" << std::endl;
    //    return -1;
    //}

    //std::cout << "superpixel_labels size: " << result.superpixel_labels.cols
    //    << "x" << result.superpixel_labels.rows << std::endl;

    //// Теперь создаем source_infos
    //std::vector<SourceGroupInfo> source_infos;
    //for (size_t i = 0; i < result.groups.size(); ++i) {
    //    SourceGroupInfo info;
    //    info.group = result.groups[i];
    //    info.superpixel_id = result.group_to_superpixel[i];

    //    std::cout << "Group " << i << " -> superpixel " << info.superpixel_id << std::endl;

    //    info.superpixel_mask = analyzer.getSuperpixelMask(
    //        result.superpixel_labels,
    //        info.superpixel_id
    //    );

    //    if (info.superpixel_mask.empty()) {
    //        std::cerr << "Warning: Empty mask for superpixel " << info.superpixel_id << std::endl;
    //    }

    //    source_infos.push_back(info);
    //}

    //std::cout << "Created " << source_infos.size() << " source infos" << std::endl;

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

    //std::vector<PlacedGroup> placed_groups = synthesizer.synthesizePlacement(
    //    source_groups,
    //    output_width, output_height,
    //    synth_density,
    //    synth_angle_variation,
    //    synth_scale_variation
    //);

    std::vector<PlacedGroup> placed_groups = synthesizer.synthesizePlacement(
       ebpt_model.getEdgeGroups(), output_width, output_height, density, synth_angle_variation, synth_scale_variation);

    // Визуализируем размещение
    cv::Mat placement_map = synthesizer.drawPlacementMap(
        placed_groups, output_width, output_height
    );
    ImageDisplay::setPartFinalVisualization(placement_map, ImageDisplay::placement);

    // Заполнение пикселей
    PixelSynthesis pixel_synthesis;
    pixel_synthesis.setRandomSeed(123);
    pixel_synthesis.setPatchSelectionMethod(1);

    //cv::Mat output_texture = pixel_synthesis.fillPixels(
    //    input_image,
    //    groups,
    //    placed_groups,
    //    output_width, output_height
    //);
    // Заполнение пикселей с масками
    cv::Mat output_texture = pixel_synthesis.PatchCopy(
        input_image, ebpt_model.getEdgeGroups(), placed_groups, output_width, output_height);

    ImageDisplay::setPartFinalVisualization(output_texture, ImageDisplay::output);

    // Создаем итоговую визуализацию
    //int viz_width = 1200;
    //int viz_height = 800;
    //cv::Mat final_visualization = cv::Mat::zeros(viz_height, viz_width, CV_8UC3);
    //final_visualization.setTo(cv::Scalar(30, 30, 30));

    //cv::Mat input_resized;
    //cv::resize(input_image, input_resized, cv::Size(350, 350));
    //input_resized.copyTo(final_visualization(cv::Rect(50, 50, 350, 350)));
    //cv::rectangle(final_visualization, cv::Rect(50, 50, 350, 350),
    //    cv::Scalar(255, 255, 255), 2);
    //cv::putText(final_visualization, "Input Texture",
    //    cv::Point(60, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
    //    cv::Scalar(255, 255, 255), 2);

    //cv::Mat edges_resized;
    //cv::resize(edges_visualization, edges_resized, cv::Size(350, 350));
    //edges_resized.copyTo(final_visualization(cv::Rect(450, 50, 350, 350)));
    //cv::rectangle(final_visualization, cv::Rect(450, 50, 350, 350),
    //    cv::Scalar(255, 255, 255), 2);
    //cv::putText(final_visualization, "Detected Edges (" + std::to_string(edges.size()) + ")",
    //    cv::Point(460, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
    //    cv::Scalar(255, 255, 255), 2);

    //cv::Mat groups_resized;
    //cv::resize(groups_visualization, groups_resized, cv::Size(350, 350));
    //groups_resized.copyTo(final_visualization(cv::Rect(50, 450, 350, 350)));
    //cv::rectangle(final_visualization, cv::Rect(50, 450, 350, 350),
    //    cv::Scalar(255, 255, 255), 2);
    //cv::putText(final_visualization, "Edge Groups (" + std::to_string(ebpt_model.getEdgeGroups().size()) + ")",
    //    cv::Point(60, 440), cv::FONT_HERSHEY_SIMPLEX, 0.7,
    //    cv::Scalar(255, 255, 255), 2);

    //cv::Mat output_resized;
    //cv::resize(output_texture, output_resized, cv::Size(350, 350));
    //output_resized.copyTo(final_visualization(cv::Rect(450, 450, 350, 350)));
    //cv::rectangle(final_visualization, cv::Rect(450, 450, 350, 350),
    //    cv::Scalar(255, 255, 255), 2);
    //cv::putText(final_visualization, "Output Texture",
    //    cv::Point(460, 440), cv::FONT_HERSHEY_SIMPLEX, 0.7,
    //    cv::Scalar(255, 255, 255), 2);

    //cv::Mat map_resized;
    //cv::resize(placement_map, map_resized, cv::Size(350, 350));
    //map_resized.copyTo(final_visualization(cv::Rect(825, 50, 350, 350)));
    //cv::rectangle(final_visualization, cv::Rect(825, 50, 350, 350),
    //    cv::Scalar(255, 255, 255), 2);
    //cv::putText(final_visualization, "Placement Map",
    //    cv::Point(835, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
    //    cv::Scalar(255, 255, 255), 2);

    /*cv::Rect info_rect(825, 450, 350, 350);
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
        "  Groups created: " + std::to_string(ebpt_model.getEdgeGroups().size()),
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
        cv::Scalar(255, 255, 255), 1);*/

    //cv::imwrite("images/final_report.png", final_visualization);

    //cv::namedWindow("Input Texture", cv::WINDOW_NORMAL);
    //cv::imshow("Input Texture", input_image);

    //cv::namedWindow("Detected Edges", cv::WINDOW_NORMAL);
    //cv::imshow("Detected Edges", edges_visualization);

    //cv::namedWindow("Edge Groups", cv::WINDOW_NORMAL);
    //cv::imshow("Edge Groups", groups_visualization);

    //cv::namedWindow("Placement Map", cv::WINDOW_NORMAL);
    //cv::imshow("Placement Map", placement_map);

    //cv::namedWindow("Output Texture", cv::WINDOW_NORMAL);
    //cv::imshow("Output Texture", output_texture);

    //cv::namedWindow("Final Report", cv::WINDOW_NORMAL);
    //cv::imshow("Final Report", final_visualization);

    //ImageDisplay::showFinalVisualization();
    bool show_edges = true;
    bool show_groups = true;
    bool running = true;

    while (running) {
        int key = cv::waitKey(100);

        switch (key) {
        case 'q':
        case 'Q':
        case 27:
            running = false;
            break;

        case 's':
        case 'S':
            cv::imwrite("images/current_input.png", input_image);
            //cv::imwrite("images/current_edges.png", edges_visualization);
            //cv::imwrite("images/current_groups.png", groups_visualization);
            cv::imwrite("images/current_placement.png", placement_map);
            cv::imwrite("images/current_output.png", output_texture);
            break;

        case 'к':
        case 'К':
        case 'r':
        case 'R':
            placed_groups = synthesizer.synthesizePlacement(
                ebpt_model.getEdgeGroups(),
                output_width, output_height,
                density, synth_angle_variation, synth_scale_variation
            );

            placement_map = synthesizer.drawPlacementMap(
                placed_groups, output_width, output_height
            );
            ImageDisplay::setPartFinalVisualization(placement_map, ImageDisplay::placement);

            output_texture = pixel_synthesis.PatchCopy(
                input_image, ebpt_model.getEdgeGroups(), placed_groups,
                output_width, output_height
            );
            ImageDisplay::setPartFinalVisualization(output_texture, ImageDisplay::output);

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

        }
    }

    cv::destroyAllWindows();
    return 0;
}