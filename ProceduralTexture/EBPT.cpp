#include "EBPT.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include<opencv2/opencv.hpp>

namespace EBPTns {

    void EBPT::addEdgeGroup(const SourceGroupInfo& group) {
        edge_groups_.push_back(group);
    }

    void EBPT::clear() {
        edge_groups_.clear();
    }

}