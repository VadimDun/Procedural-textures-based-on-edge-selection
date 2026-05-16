#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <random>

#include "Edge.h"
#include "EdgeGroup.h"
#include "PlacedGroup.h"

using namespace EBPTns;
static const std::string IMAGE_FOLDER = "images/";

class ImageDisplay {
public:

    enum PartFinalVis:uchar { input, edges, groups, output, placement};
    static void save(const std::string& path, const cv::Mat& mat);
    static void show(const std::string& nameWindow, const cv::Mat& mat);
    static void saveAndShowWithSize(const std::string& path, const std::string& nameWindow, const cv::Mat& mat, const cv::Size& size);
    static void saveAndShow(const std::string& path, const std::string& nameWindow, const cv::Mat& mat);

    static cv::Mat visualizeGroups(const cv::Mat& image, const std::vector<SourceGroupInfo>& groups);
    static cv::Mat visualizeEdges(const cv::Mat& image, const std::vector<Edge>& edges);
    static cv::Mat visualizeSuperpixels(const cv::Mat& image, const cv::Mat& labels);

    static void visualiseSPWithEdges(const cv::Mat& image, const cv::Mat& spVis, const cv::Mat& edgeVis);

    static cv::Mat drawPlacementMap(
        const std::vector<PlacedGroup>& placed_groups,
        const cv::Size& size);

    static void showOccupancyMap(const cv::Mat& occupancy_map, const std::string& title = "Occupancy Map");

    static cv::Mat visualizeSuperpixelLabels(const cv::Mat& image, const cv::Mat& labels);

private:
    static void drawArrow(cv::Mat& img, const cv::Point& end, float angle, const cv::Scalar& color);
};