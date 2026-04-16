#pragma once

#include "EdgeGroup.h"
#include <vector>
#include <opencv2/core.hpp>
#include <random>

namespace EBPTns {

    enum class ScaleLevel: uint8_t {
        LARGE,
        MEDIUM,
        SMALL,
    };

    inline std::string scaleLevelToString(ScaleLevel level) {
        switch (level) {
        case ScaleLevel::LARGE:  return "LARGE";
        case ScaleLevel::MEDIUM: return "MEDIUM";
        case ScaleLevel::SMALL:  return "SMALL";
        default: return "UNKNOWN";
        }
    }

    struct SourceGroupInfo {
        EdgeGroup group;
        int superpixel_id;
        cv::Mat mask;
        std::vector<cv::Point> hull;

        ScaleLevel scale_level;

        SourceGroupInfo() {}

        SourceGroupInfo& operator=(const SourceGroupInfo& other) {
            if (this != &other) {
                group = other.group;
                superpixel_id = other.superpixel_id;
                other.mask.copyTo(mask);  // Deep copy for cv::Mat
                hull = other.hull;
                scale_level = other.scale_level;
            }
            return *this;
        }

        //SourceGroupInfo& operator=(SourceGroupInfo&& other) noexcept {
        //    if (this != &other) {
        //        group = std::move(other.group);
        //        superpixel_id = other.superpixel_id;
        //        mask = std::move(other.mask);  // cv::Mat has move semantics
        //        hull = std::move(other.hull);
        //        scale_level = other.scale_level;
        //    }
        //    return *this;
        //}
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