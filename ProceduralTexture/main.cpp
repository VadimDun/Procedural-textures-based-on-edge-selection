#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Edge.h"
#include "EdgeGroup.h"
#include "SourceGroupInfo.h"
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

    //if (image.cols > 512 || image.rows > 512) {
        //cv::resize(image, image, cv::Size(512, 512));
    //}

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

    ImageDisplay::saveAndShowWithSize("input_texture.png", "Input Texture", input_image, cv::Size(input_image.cols, input_image.rows));

    // Анализ текстуры
    TextureAnalysis analyzer;

    if (use_real_texture) {
        analyzer.setMinEdgeLength(15);

        int regionSize = 120;
        double threshold = 0.1;
        analyzer.setSuperpixelParams(regionSize, 10.0f, threshold);

    }
    else {
        analyzer.setMinEdgeLength(10);

        int regionSize = 100;
        double threshold = 0.25;
        analyzer.setSuperpixelParams(regionSize, 10.0f, threshold);
    }

    std::cout << "\n\n---------------------analyzeTextureWithSuperpixelsStructured---------------------\n\n" << std::endl;
    auto result = analyzer.analyzeTexture(input_image, MODEL_PATH);
    if (!result.isValid()) { return 1; }

    /////////////////////////////
    // Бины
    /////////////////////////////

    //ImageDisplay::visualizeEdgeBins(input_image, result.modelEBPT.getEdgeGroups(), "images/edge_bins_structured.png");
    //ImageDisplay::visualizeBinDistribution(result.modelEBPT.getEdgeGroups(), "images/bin_distribution.png");

    //////////////////////////////////

    /////////////////////////////
    // Синтез размещения
    /////////////////////////////
    cv::Size outSize;
    if (use_real_texture) {
        if (argc > 3) {
            std::string arg3 = argv[3];
            std::string arg4 = argv[4];
            try {
                int height = std::stoi(arg3);
                int width = std::stoi(arg4);
                outSize.height = height;
                outSize.width = width;
            }
            catch (...) {
                outSize.height = input_image.rows * 2;
                outSize.width = input_image.cols * 2;
            }
        }
        else {
            outSize.height = input_image.rows * 2;
            outSize.width = input_image.cols * 2;
        }
    }
    else {
        outSize.width = 800;
        outSize.height = 600;
    }

    std::cout << "---------------------------- OUTPUT SIZE=" << outSize << " ---------------------------------------------";

    float scale = 0.5f;
    float density = 1.3f;
    float angle_spread = 0.1f;

    bool enable_rotation = true;

    if (use_real_texture) {
        scale = 0.2f;
        density = 0.9f;

        if (argc > 2) {
            std::string arg2 = argv[2];
            if (arg2 == "0" || arg2 == "false" || arg2 == "off" || arg2 == "no") {
                enable_rotation = false;
                std::cout << "\n\n---------------------Rotation disabled---------------------\n\n" << std::endl;
            }
            else if (arg2 == "1" || arg2 == "true" || arg2 == "on" || arg2 == "yes") {
                enable_rotation = true;
                std::cout << "\n\n---------------------Rotation enabled---------------------\n\n" << std::endl;
            }
            else {
                std::cout << "Unknown rotation parameter: " << arg2
                    << ". Using default (rotation enabled)" << std::endl;
            }
        }

        if (enable_rotation)
            angle_spread = 0.1f;
        else angle_spread = 0.0f;
    }

    TextureSynthesis synthesizer(outSize, enable_rotation);
    synthesizer.setRandomSeed(42);
    synthesizer.setAvoidOverlap(true);
    
    const auto& source_groups = result.source_groups;

    std::vector<PlacedGroup> placed_groups;
    std::cout << "\n\n---------------------synthesizeHierarchicalPlacement---------------------\n\n" << std::endl;

    placed_groups = synthesizer.synthesizeHierarchicalPlacement(
        input_image,
        source_groups
    );


    // Визуализируем размещение
    cv::Mat placement_map = ImageDisplay::drawPlacementMap(
        placed_groups, outSize
    );
    ImageDisplay::saveAndShowWithSize("placement_map.png", "Placement Map", placement_map, outSize);
    //placement_map.~Mat();
    // Заполнение пикселей
    PixelSynthesis pixel_synthesis;

    std::cout << "\n\n---------------------fillPixels---------------------\n\n" << std::endl;
    // Заполнение пикселей с масками
    cv::Mat output_texture = pixel_synthesis.fillPixels(
        input_image, source_groups, placed_groups, outSize);
    ImageDisplay::saveAndShowWithSize("output_texture.png", "Output Texture", output_texture, outSize);

    bool running = true;
    while (running) {
        int key = cv::waitKey(100);

        switch (key) {
        case 'q':
        case 'Q':
        case 27:
            running = false;
            break;
        case 'к':
        case 'К':
        case 'r':
        case 'R':
            placed_groups = synthesizer.synthesizeHierarchicalPlacement(
                input_image,
                source_groups
            );


            placement_map = ImageDisplay::drawPlacementMap(
                placed_groups, outSize
            );
            ImageDisplay::saveAndShowWithSize("placement_map.png", "Placement Map", placement_map, outSize);

            output_texture = pixel_synthesis.fillPixels(
                input_image, source_groups, placed_groups,
                outSize
            );
            ImageDisplay::saveAndShowWithSize("output_texture.png", "Output Texture", output_texture, outSize);

            cv::imshow("Placement Map", placement_map);
            cv::imshow("Output Texture", output_texture);
        break;
        case 'n':
        case 'N':
        {
            unsigned int new_seed = std::random_device{}();
            synthesizer.setRandomSeed(new_seed);
        }
        break;

        }
    }

    cv::destroyAllWindows();
    return 0;
}