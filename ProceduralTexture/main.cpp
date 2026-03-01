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

using namespace EBPTns;

cv::Mat createTestImage(int width = 400, int height = 400) {
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

cv::Mat loadRealTexture(const std::string& path) {
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

    std::vector<Edge> edges = analyzer.extractEdges(input_image);
    cv::Mat edges_visualization = analyzer.visualizeEdges(input_image, edges);
    cv::imwrite("images/edges_detected.png", edges_visualization);

    std::vector<EdgeGroup> groups = analyzer.groupEdges(edges);
    cv::Mat groups_visualization = analyzer.visualizeGroups(input_image, groups);
    cv::imwrite("images/groups_detected.png", groups_visualization);

    // Создаем EBPT модель
    EBPT ebpt_model(input_image);

    for (const auto& group : groups) {
        ebpt_model.addEdgeGroup(group);
    }

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