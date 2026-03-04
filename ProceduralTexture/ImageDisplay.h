#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Edge.h"
#include "EdgeGroup.h"
#include "EBPT.h"

using namespace EBPTns;
static const std::string IMAGE_FOLDER = "images/";

class ImageDisplay {
public:

    enum PartFinalVis:uchar { input, edges, groups, output, placement};
    static void save(const std::string& path, const cv::Mat& mat);
    static void show(const std::string& nameWindow, const cv::Mat& mat);
    static void saveAndShow(const std::string& path, const std::string& nameWindow, const cv::Mat& mat);

    static void initFinalVisualization();
    static void setPartFinalVisualization(const cv::Mat& mat, PartFinalVis part);
    static void showFinalVisualization();

    static cv::Mat visualizeGroups(const cv::Mat& image, const std::vector<SourceGroupInfo>& groups);
    static cv::Mat visualizeEdges(const cv::Mat& image, const std::vector<Edge>& edges);
    static cv::Mat visualizeSuperpixels(const cv::Mat& image, const cv::Mat& labels);

    static void visualiseSPWithEdges(const cv::Mat& image, const cv::Mat& spVis, const cv::Mat& edgeVis);

    static void visualizeChainCode(const EBPTns::Edge& edge, cv::Mat& image,
        const cv::Scalar& color = cv::Scalar(0, 255, 0),
        bool show_angle = true);

    static void visualizeAllChainCodes(const std::vector<EBPTns::Edge>& edges,
        cv::Mat& image,
        const std::string& filename = "images/chain_code_viz.png");

    static void visualizeAnglesOnly(const std::vector<EBPTns::Edge>& edges,
        const cv::Mat& background,
        const std::string& filename = "images/angles_only.png");

    static void visualizeEdgeBins(const cv::Mat& input_image,
        const std::vector<EBPTns::Edge>& edges,
        const std::string& filename = "images/edge_bins.png");

    static void visualizeBinDistribution(const std::vector<EBPTns::Edge>& edges,
        const std::string& filename = "images/bin_distribution.png");

private:

    static cv::Mat final_visualization;
};