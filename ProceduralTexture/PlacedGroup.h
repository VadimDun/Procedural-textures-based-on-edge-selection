#pragma once

#include "EdgeGroup.h"
#include <opencv2/imgproc.hpp>

namespace EBPTns {


    struct PlacedGroup {
        EdgeGroup group;           // Трансформированная группа
        int source_index;          // Индекс в исходном массиве групп
        int superpixel_id;
        float scale_factor;     
        float rotation_angle;   
        cv::Point2f translation;

        PlacedGroup() : source_index(-1), scale_factor(1.0f), rotation_angle(0.0f), superpixel_id(-1) {}

        PlacedGroup(const EdgeGroup& g, int idx, int sp_id, float scale = 1.0f,
            float angle = 0.0f, const cv::Point2f& trans = cv::Point2f(0, 0))
            : group(g), source_index(idx), superpixel_id(sp_id), scale_factor(scale),
            rotation_angle(angle), translation(trans) {
        }

        bool isValid() const { return source_index >= 0; }

        cv::Mat getTransformationMatrix() const {
            cv::Mat scale_mat = (cv::Mat_<float>(2, 3) <<
                scale_factor, 0, 0,
                0, scale_factor, 0);

            cv::Point2f center = group.getCenter();
            cv::Mat rotation_mat = cv::getRotationMatrix2D(
                cv::Point2f(0, 0),
                rotation_angle * 180.0f / CV_PI,
                1.0);

            cv::Mat transform = scale_mat * rotation_mat;

            transform.at<float>(0, 2) += translation.x;
            transform.at<float>(1, 2) += translation.y;

            return transform;
        }
    };

}