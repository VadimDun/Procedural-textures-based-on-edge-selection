#pragma once

#include "EdgeGroup.h"
#include <vector>
#include <opencv2/core.hpp>
#include <random>

namespace EBPTns {

    struct SourceGroupInfo {
        EdgeGroup group;
        int superpixel_id;
        cv::Mat superpixel_mask;
    };

    class EBPT {
    public:
        EBPT() = default;
        EBPT(const cv::Mat& input_image);

        void generateTexture(cv::Mat& output, int width, int height);

        void setScale(float scale) { scale_ = scale; }
        void setDensity(float density) { density_ = density; }
        void setAngleSpread(float spread) { angle_spread_ = spread; }

        const cv::Mat& getInputImage() const { return input_image_; }
        const std::vector<SourceGroupInfo>& getEdgeGroups() const { return edge_groups_; }
        int getNumGroups() const { return static_cast<int>(edge_groups_.size()); }

        void clear();
        bool isEmpty() const { return edge_groups_.empty(); }

        void addEdgeGroup(const SourceGroupInfo& group);

    private:
        cv::Mat input_image_;               
        std::vector<SourceGroupInfo> edge_groups_;

        float scale_ = 1.0f;
        float density_ = 1.0f;                    // Плотность (0.0-1.0)
        float angle_spread_ = 0.5f;               // Разброс углов (0.0-1.0)

        mutable std::mt19937 rng_;

        void initializeRNG();
        cv::Point2f getRandomPosition(int width, int height) const;
        float getRandomAngle() const;
        float getRandomScale() const;

    };

}