#include "EBPT.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include<opencv2/opencv.hpp>

namespace EBPTns {

    EBPT::EBPT(const cv::Mat& input_image)
        : input_image_(input_image) {
        initializeRNG();
    }

    void EBPT::initializeRNG() {
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }

    void EBPT::addEdgeGroup(const EdgeGroup& group) {
        edge_groups_.push_back(group);
    }

    void EBPT::clear() {
        edge_groups_.clear();
        input_image_ = cv::Mat();
    }

    cv::Point2f EBPT::getRandomPosition(int width, int height) const {
        if (width <= 0 || height <= 0) {
            return cv::Point2f(0, 0);
        }
        float margin = 50.0f;
        std::uniform_real_distribution<float> dist_x(margin,
            std::max(margin, static_cast<float>(width) - margin));
        std::uniform_real_distribution<float> dist_y(margin,
            std::max(margin, static_cast<float>(height) - margin));
        return cv::Point2f(dist_x(rng_), dist_y(rng_));
    }

    float EBPT::getRandomAngle() const {
        std::uniform_real_distribution<float> dist(-angle_spread_ * CV_PI,
            angle_spread_ * CV_PI);
        return dist(rng_);
    }

    float EBPT::getRandomScale() const {
        std::uniform_real_distribution<float> dist(0.7f * scale_, 1.3f * scale_);
        return std::max(0.1f, dist(rng_));
    }

    void EBPT::generateTexture(cv::Mat& output, int width, int height) {
        if (edge_groups_.empty() || input_image_.empty()) {
            output = cv::Mat::zeros(height, width, CV_8UC3);
            return;
        }
        if (width <= 0 || height <= 0) {
            width = input_image_.cols * 2;
            height = input_image_.rows * 2;
        }
        output = cv::Mat::zeros(height, width, input_image_.type());
        int processed_groups = 0;
        for (const auto& group : edge_groups_) {
            cv::Rect bbox = group.getBoundingBox();
            if (bbox.x < 0) bbox.x = 0;
            if (bbox.y < 0) bbox.y = 0;
            if (bbox.x + bbox.width > width) bbox.width = width - bbox.x;
            if (bbox.y + bbox.height > height) bbox.height = height - bbox.y;
            if (bbox.width <= 0 || bbox.height <= 0) {
                continue;
            }
            if (bbox.width > input_image_.cols || bbox.height > input_image_.rows) {
                cv::Mat resized_patch;
                cv::resize(input_image_, resized_patch, bbox.size());
                resized_patch.copyTo(output(bbox));
            }
            else {
                int patch_x = std::uniform_int_distribution<int>(
                    0, std::max(0, input_image_.cols - bbox.width - 1))(rng_);
                int patch_y = std::uniform_int_distribution<int>(
                    0, std::max(0, input_image_.rows - bbox.height - 1))(rng_);
                cv::Rect source_rect(patch_x, patch_y, bbox.width, bbox.height);
                cv::Mat patch = input_image_(source_rect).clone();
                patch.copyTo(output(bbox));
            }
            processed_groups++;
        }
        if (processed_groups == 0) {
            cv::Scalar avg_color = cv::mean(input_image_);
            output = cv::Mat(height, width, input_image_.type(), avg_color);
            cv::Mat noise = cv::Mat(height, width, input_image_.type());
            cv::randn(noise, 0, 30);
            output += noise;
        }
    }

}