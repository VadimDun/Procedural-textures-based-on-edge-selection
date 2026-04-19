#pragma once

#include "EdgeGroup.h"
#include <vector>
#include <opencv2/core.hpp>

namespace EBPTns {

    enum class ScaleLevel : uint8_t {
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
        uint8_t superpixel_id;
        std::vector<cv::Point> hull;

        ScaleLevel scale_level;

        SourceGroupInfo() {}

        SourceGroupInfo& operator=(const SourceGroupInfo& other) {
            if (this != &other) {
                group = other.group;
                superpixel_id = other.superpixel_id;
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

}