#include "ImageDisplay.h"

void ImageDisplay::save(const std::string& path, const cv::Mat& mat) {
    cv::imwrite(IMAGE_FOLDER + path, mat);
}

void ImageDisplay::show(const std::string& nameWindow, const cv::Mat& mat) {
    cv::namedWindow(nameWindow, cv::WINDOW_NORMAL);
    cv::imshow(nameWindow, mat);
}

void ImageDisplay::saveAndShow(const std::string& path, const std::string& nameWindow, const cv::Mat& mat) {
    save(path, mat);
    show(nameWindow, mat);
}

void ImageDisplay::saveAndShowWithSize(const std::string& path, const std::string& nameWindow, const cv::Mat& mat, const cv::Size& size) {
    save(path, mat);
    cv::namedWindow(nameWindow, cv::WINDOW_NORMAL);
    cv::resizeWindow(nameWindow, size.width / 2, size.height / 2);
    cv::imshow(nameWindow, mat);
}

////////////////////////////// 
// Визуализация
//////////////////////////////

cv::Mat ImageDisplay::visualizeEdges(const cv::Mat& image,
    const std::vector<Edge>& edges) {
    cv::Mat visualization;
    if (image.channels() == 1) {
        cv::cvtColor(image, visualization, cv::COLOR_GRAY2BGR);
    }
    else {
        visualization = image.clone();
    }
    for (const auto& edge : edges) {
        const auto& points = edge.getPoints();
        if (points.size() < 2) continue;
        for (size_t i = 1; i < points.size(); ++i) {
            cv::line(visualization, points[i - 1], points[i],
                cv::Scalar(0, 255, 0),
                2);
        }
        cv::Point center = cv::Point(
            static_cast<int>(edge.getCenter().x),
            static_cast<int>(edge.getCenter().y)
        );
        cv::circle(visualization, center, 3,
            cv::Scalar(0, 0, 255), -1);
    }
    return visualization;
}

cv::Mat ImageDisplay::visualizeGroups(const cv::Mat& image, const std::vector<SourceGroupInfo>& groups) {
    cv::Mat visualization;
    if (image.channels() == 1) {
        cv::cvtColor(image, visualization, cv::COLOR_GRAY2BGR);
    }
    else {
        visualization = image.clone();
    }

    std::vector<cv::Scalar> group_colors = {
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(128, 0, 0),
        cv::Scalar(0, 128, 0),
        cv::Scalar(0, 0, 128),
        cv::Scalar(128, 128, 0)
    };

    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
        const SourceGroupInfo& group_info = groups[group_idx];
        const EdgeGroup& group = group_info.group;
        cv::Scalar color = group_colors[group_idx % group_colors.size()];

        // Рисуем ребра группы
        for (const auto& edge : group.getEdges()) {
            const auto& points = edge.getPoints();
            if (points.size() < 2) continue;
            for (size_t i = 1; i < points.size(); ++i) {
                cv::line(visualization, points[i - 1], points[i],
                    color, 2);
            }
        }

        // Рисуем центр группы
        cv::Point group_center = cv::Point(
            static_cast<int>(group.getCenter().x),
            static_cast<int>(group.getCenter().y)
        );
        cv::circle(visualization, group_center, 5, color, -1);

        // Рисуем выпуклую оболочку
        const std::vector<cv::Point>& hull = group_info.hull;

        if (!hull.empty() && hull.size() >= 3) {
            // Рисуем контур выпуклой оболочки
            for (size_t i = 0; i < hull.size(); ++i) {
                cv::line(visualization, hull[i], hull[(i + 1) % hull.size()],
                    color, 2);
            }

            // Заливка с прозрачностью
            cv::Mat overlay = visualization.clone();
            cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{hull},
                cv::Scalar(color[0], color[1], color[2]));
            cv::addWeighted(overlay, 0.3, visualization, 0.7, 0, visualization);

            // Вычисляем bounding box для размещения текста
            cv::Rect bbox = cv::boundingRect(hull);
            std::string label = "G" + std::to_string(group_idx + 1);
            cv::putText(visualization, label,
                cv::Point(group_center.x, group_center.y - 5),
                //cv::Point(bbox.x, bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2);
        }
    }

    return visualization;
}

cv::Mat ImageDisplay::visualizeSuperpixels(const cv::Mat& image, const cv::Mat& labels) {
    cv::Mat visualization;

    if (image.channels() == 1) {
        cv::cvtColor(image, visualization, cv::COLOR_GRAY2BGR);
    }
    else {
        visualization = image.clone();
    }

    if (labels.empty()) {
        return visualization;
    }

    std::unordered_map<int, cv::Scalar> sp_colors;
    cv::RNG rng(12345);

    // Рисуем границы суперпикселей
    cv::Mat contour_mask = cv::Mat::zeros(labels.size(), CV_8UC1);

    // Проходим по всем пикселям и находим границы
    for (int y = 1; y < labels.rows - 1; ++y) {
        for (int x = 1; x < labels.cols - 1; ++x) {
            int current = labels.at<int>(y, x);

            // Проверяем соседей
            if (current != labels.at<int>(y - 1, x) ||
                current != labels.at<int>(y + 1, x) ||
                current != labels.at<int>(y, x - 1) ||
                current != labels.at<int>(y, x + 1)) {
                contour_mask.at<uchar>(y, x) = 255;
            }
        }
    }

    // Рисуем границы красным
    visualization.setTo(cv::Scalar(0, 0, 255), contour_mask);

    // Добавляем центры суперпикселей
    std::unordered_map<int, cv::Point> centers;
    std::unordered_map<int, int> counts;

    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            int sp_id = labels.at<int>(y, x);
            centers[sp_id].x += x;
            centers[sp_id].y += y;
            counts[sp_id]++;
        }
    }

    for (auto& [sp_id, center] : centers) {
        if (counts[sp_id] > 0) {
            center.x /= counts[sp_id];
            center.y /= counts[sp_id];

            if (sp_colors.find(sp_id) == sp_colors.end()) {
                sp_colors[sp_id] = cv::Scalar(rng.uniform(0, 255),
                    rng.uniform(0, 255),
                    rng.uniform(0, 255));
            }

            cv::circle(visualization, center, 3, sp_colors[sp_id], -1);

            std::string sp_text = std::to_string(sp_id);
            cv::putText(visualization, sp_text,
                cv::Point(center.x + 5, center.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, sp_colors[sp_id], 1);
        }
    }

    return visualization;
}

void ImageDisplay::visualiseSPWithEdges(const cv::Mat& image, const cv::Mat& spVis, const cv::Mat& edgeVis) {
    cv::Mat composite = image.clone();
    cv::addWeighted(composite, 0.7, spVis, 0.3, 0, composite);
    cv::addWeighted(composite, 1.0, edgeVis, 0.5, 0, composite);
    saveAndShow("superpixels_with_edges.png", "sp with edges", composite);
}

cv::Mat ImageDisplay::drawPlacementMap(
    const std::vector<PlacedGroup>& placed_groups,
    const cv::Size& size) {

    cv::Mat placement_map = cv::Mat::zeros(size.height, size.width, CV_8UC3);

    if (placed_groups.empty()) {
        cv::putText(placement_map, "No groups placed",
            cv::Point(10, 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.7,
            cv::Scalar(255, 255, 255), 2);
        return placement_map;
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> color_dist(0, 255);

    for (size_t i = 0; i < placed_groups.size(); ++i) {
        const auto& placed = placed_groups[i];

        cv::Scalar color = cv::Scalar(color_dist(rng), color_dist(rng), color_dist(rng));
        cv::Point center(static_cast<int>(placed.position.x),
            static_cast<int>(placed.position.y));

        // Рисуем центр
        cv::circle(placement_map, center, 8, color, -1);
        cv::circle(placement_map, center, 3, cv::Scalar(255, 255, 255), -1);

        // Рисуем hull
        if (!placed.hull.empty() && placed.hull.size() >= 3) {
            // Контур
            for (size_t j = 0; j < placed.hull.size(); ++j) {
                cv::line(placement_map,
                    placed.hull[j],
                    placed.hull[(j + 1) % placed.hull.size()],
                    color, 2);
            }

            // Заливка
            cv::Mat overlay = placement_map.clone();
            std::vector<std::vector<cv::Point>> hull_contour = { placed.hull };
            cv::fillPoly(overlay, hull_contour, cv::Scalar(color[0], color[1], color[2]));
            cv::addWeighted(overlay, 0.2, placement_map, 0.8, 0, placement_map);

            // Информация с углом для отладки
            cv::Rect bbox = cv::boundingRect(placed.hull);
            std::string info = "G" + std::to_string(placed.source_index) +
                " A:" + std::to_string(static_cast<int>(placed.rotation_angle * 180.0 / CV_PI)) + "deg";
            cv::putText(placement_map, info,
                cv::Point(center.x, center.y - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1);
        }

        // Рисуем направление
        float angle = placed.rotation_angle;
        int line_length = 40;  // Увеличим для наглядности
        cv::Point direction(
            center.x + static_cast<int>(line_length * std::cos(angle)),
            center.y + static_cast<int>(line_length * std::sin(angle))
        );
        cv::line(placement_map, center, direction, color, 3);

        // Рисуем стрелку
        drawArrow(placement_map, direction, angle, color);

        // Рисуем маленькую окружность в направлении для наглядности
        cv::circle(placement_map, direction, 4, color, -1);

        // Номер группы
        std::string label = std::to_string(i + 1);
        cv::putText(placement_map, label,
            cv::Point(center.x + 10, center.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(255, 255, 255), 2);
    }

    // Заголовок
    std::string title = "Placement Map: " + std::to_string(placed_groups.size()) + " groups";
    cv::putText(placement_map, title,
        cv::Point(10, 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2);

    return placement_map;
}

void ImageDisplay::showOccupancyMap(const cv::Mat& occupancy_map, const std::string& title) {
    if (occupancy_map.empty()) return;

    cv::Mat display;
    cv::applyColorMap(occupancy_map, display, cv::COLORMAP_JET);
    show(title, display);
    cv::waitKey(1);
}

void ImageDisplay::drawArrow(cv::Mat& img, const cv::Point& end, float angle, const cv::Scalar& color) {
    double arrow_angle = angle;
    int arrow_length = 12;

    cv::Point arrow_tip = end;
    cv::Point arrow_left(
        arrow_tip.x - static_cast<int>(arrow_length * std::cos(arrow_angle - CV_PI / 6)),
        arrow_tip.y - static_cast<int>(arrow_length * std::sin(arrow_angle - CV_PI / 6))
    );
    cv::Point arrow_right(
        arrow_tip.x - static_cast<int>(arrow_length * std::cos(arrow_angle + CV_PI / 6)),
        arrow_tip.y - static_cast<int>(arrow_length * std::sin(arrow_angle + CV_PI / 6))
    );

    cv::line(img, arrow_tip, arrow_left, color, 2);
    cv::line(img, arrow_tip, arrow_right, color, 2);
}

cv::Mat ImageDisplay::visualizeSuperpixelLabels(const cv::Mat& image, const cv::Mat& labels) {
    cv::Mat visualization;

    if (image.channels() == 1) {
        cv::cvtColor(image, visualization, cv::COLOR_GRAY2BGR);
    }
    else {
        visualization = image.clone();
    }

    if (labels.empty()) {
        return visualization;
    }

    // Генерируем уникальные цвета для каждого суперпикселя
    std::unordered_map<int, cv::Scalar> sp_colors;
    cv::RNG rng(12345);

    // Находим все уникальные метки суперпикселей
    std::set<int> unique_labels;
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            unique_labels.insert(labels.at<int>(y, x));
        }
    }

    // Создаем цветную карту для каждого суперпикселя
    for (int label : unique_labels) {
        // Генерируем яркий, насыщенный цвет
        cv::Scalar color(
            rng.uniform(100, 255),
            rng.uniform(100, 255),
            rng.uniform(100, 255)
        );
        sp_colors[label] = color;
    }

    // Закрашиваем каждый суперпиксель своим цветом с прозрачностью
    cv::Mat colored_labels = cv::Mat::zeros(labels.size(), CV_8UC3);
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            int label = labels.at<int>(y, x);
            colored_labels.at<cv::Vec3b>(y, x) = cv::Vec3b(
                sp_colors[label][0],
                sp_colors[label][1],
                sp_colors[label][2]
            );
        }
    }

    // Накладываем цветные метки на исходное изображение с прозрачностью
    cv::addWeighted(visualization, 0.3, colored_labels, 0.7, 0, visualization);

    // Рисуем границы суперпикселей
    cv::Mat boundaries = cv::Mat::zeros(labels.size(), CV_8UC1);
    for (int y = 1; y < labels.rows - 1; ++y) {
        for (int x = 1; x < labels.cols - 1; ++x) {
            int current = labels.at<int>(y, x);
            // Проверяем соседей
            if (current != labels.at<int>(y - 1, x) ||
                current != labels.at<int>(y + 1, x) ||
                current != labels.at<int>(y, x - 1) ||
                current != labels.at<int>(y, x + 1)) {
                boundaries.at<uchar>(y, x) = 255;
            }
        }
    }

    // Рисуем границы белым цветом
    //visualization.setTo(cv::Scalar(255, 255, 255), boundaries);

    // Добавляем центры и номера суперпикселей
    std::unordered_map<int, cv::Point> centers;
    std::unordered_map<int, int> counts;

    // Вычисляем центры суперпикселей
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            int label = labels.at<int>(y, x);
            centers[label].x += x;
            centers[label].y += y;
            counts[label]++;
        }
    }

    // Рисуем центры и номера
    for (const auto& [label, center] : centers) {
        if (counts[label] > 0) {
            cv::Point actual_center(
                center.x / counts[label],
                center.y / counts[label]
            );

            // Рисуем центр
            cv::circle(visualization, actual_center, 4, cv::Scalar(0, 0, 0), -1);
            cv::circle(visualization, actual_center, 3, sp_colors[label], -1);

            // Добавляем номер суперпикселя
            std::string label_text = std::to_string(label);
            cv::putText(visualization, label_text,
                cv::Point(actual_center.x + 5, actual_center.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 2);
            cv::putText(visualization, label_text,
                cv::Point(actual_center.x + 5, actual_center.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        }
    }

    return visualization;
}