#include "ImageDisplay.h"

cv::Mat ImageDisplay::final_visualization;

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

void ImageDisplay::initFinalVisualization() {
    int viz_width = 1200;
    int viz_height = 800;
    final_visualization = cv::Mat::zeros(viz_height, viz_width, CV_8UC3);
    final_visualization.setTo(cv::Scalar(30, 30, 30));
}

void ImageDisplay::setPartFinalVisualization(const cv::Mat& mat, PartFinalVis part) {
    cv::Mat res;
    cv::resize(mat, res, cv::Size(350, 350));
    cv::Rect toCopy;
    cv::Point point;
    std::string text;
    std::string path;

    switch (part) {
    case PartFinalVis::input:
        toCopy = cv::Rect(50, 50, 350, 350);
        point = cv::Point(60, 40);
        text = "Input Texture";
        path = "input_texture.png";
        break;
    case PartFinalVis::edges:
        toCopy = cv::Rect(450, 50, 350, 350);
        point = cv::Point(460, 40);
        text = "Detected Edges";
        path = "edges.png";
        break;
    case PartFinalVis::groups:
        toCopy = cv::Rect(50, 450, 350, 350);
        point = cv::Point(60, 440);
        text = "Edge Groups";
        path = "groups.png";
        break;
    case PartFinalVis::output:
        toCopy = cv::Rect(450, 450, 350, 350);
        point = cv::Point(460, 440);
        text = "Output Texture";
        path = "output_texture.png";
        break;
    case PartFinalVis::placement:
        toCopy = cv::Rect(825, 50, 350, 350);
        point = cv::Point(835, 40);
        text = "Placement Map";
        path = "placement_map.png";
        break;
    }

    res.copyTo(final_visualization(toCopy));
    cv::rectangle(final_visualization, toCopy,
        cv::Scalar(255, 255, 255), 2);
    cv::putText(final_visualization, text,
        point, cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2);

    saveAndShow(path, text, mat);
}
void ImageDisplay::showFinalVisualization() {
    // TODO не знаю, стоит ли добавлять
    //cv::Rect info_rect(825, 450, 350, 350);
    //cv::rectangle(final_visualization, info_rect, cv::Scalar(50, 50, 70), -1);
    //cv::rectangle(final_visualization, info_rect, cv::Scalar(200, 200, 200), 2);

    //std::vector<std::string> info_lines = {
    //    "EBPT Texture Synthesis Report",
    //    "=============================",
    //    "Configuration:",
    //    "  Input: " + std::to_string(input_image.cols) + "x" +
    //                std::to_string(input_image.rows),
    //    "  Output: " + std::to_string(output_width) + "x" +
    //                 std::to_string(output_height),
    //    "",
    //    "Analysis Results:",
    //    "  Edges detected: " + std::to_string(edges.size()),
    //    "  Groups created: " + std::to_string(ebpt_model.getEdgeGroups().size()),
    //    "",
    //    "Synthesis Parameters:",
    //    "  Scale: " + std::to_string(scale),
    //    "  Density: " + std::to_string(density),
    //    "  Angle spread: " + std::to_string(angle_spread),
    //    "",
    //    "Placement Results:",
    //    "  Groups placed: " + std::to_string(placed_groups.size()),
    //    "  Texture type: " + std::string(use_real_texture ? "Real" : "Test")
    //};

    //for (size_t i = 0; i < info_lines.size(); ++i) {
    //    cv::putText(final_visualization, info_lines[i],
    //        cv::Point(info_rect.x + 10, info_rect.y + 30 + i * 20),
    //        cv::FONT_HERSHEY_SIMPLEX, 0.45,
    //        cv::Scalar(200, 200, 150), 1);
    //}

    //cv::putText(final_visualization, "System Info",
    //    cv::Point(info_rect.x + 10, info_rect.y + 20),
    //    cv::FONT_HERSHEY_SIMPLEX, 0.6,
    //    cv::Scalar(255, 255, 255), 1);

    save("images/final_report.png", final_visualization);
    show("Final Report", final_visualization);
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
                          cv::Scalar(color[0], color[1], color[2], 100));
             cv::addWeighted(overlay, 0.3, visualization, 0.7, 0, visualization);

            // Вычисляем bounding box для размещения текста
            cv::Rect bbox = cv::boundingRect(hull);
            std::string label = "G" + std::to_string(group_idx + 1);
            cv::putText(visualization, label,
                cv::Point(bbox.x, bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2);
        }
        else if (!group_info.superpixel_mask.empty()) {
            // Fallback: используем маску для поиска контура
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(group_info.superpixel_mask.clone(), contours,
                cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            if (!contours.empty()) {
                std::vector<cv::Point> hull_from_mask;
                cv::convexHull(contours[0], hull_from_mask);

                for (size_t i = 0; i < hull_from_mask.size(); ++i) {
                    cv::line(visualization, hull_from_mask[i],
                        hull_from_mask[(i + 1) % hull_from_mask.size()],
                        color, 2);
                }

                cv::Rect bbox = cv::boundingRect(hull_from_mask);
                std::string label = "G" + std::to_string(group_idx + 1);
                cv::putText(visualization, label,
                    cv::Point(bbox.x, bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2);
            }
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

void ImageDisplay::visualizeChainCode(const Edge& edge, cv::Mat& image,
    const cv::Scalar& color,
    bool show_angle) {
    const auto& points = edge.getPoints();
    const auto& chain = edge.getChainCode();

    if (points.size() < 2 || chain.empty()) return;

    // Рисуем стрелки цепного кода
    cv::Point prev = points[0];
    for (size_t i = 1; i < points.size(); ++i) {
        cv::line(image, prev, points[i], cv::Scalar(100, 100, 100), 1, cv::LINE_AA);

        cv::arrowedLine(image, prev, points[i],
            color,
            1,
            cv::LINE_AA,
            0,
            0.2);
        prev = points[i];
    }

    // Отмечаем начальную точку
    cv::circle(image, points[0], 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(image, "S",
        cv::Point(points[0].x + 5, points[0].y - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

    // Отмечаем конечную точку
    cv::circle(image, points.back(), 3, cv::Scalar(255, 0, 0), -1);
    cv::putText(image, "E",
        cv::Point(points.back().x + 5, points.back().y - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);

    // Визуализация направления угла (главной оси PCA)
    if (show_angle) {
        cv::Point2f center = edge.getCenter();
        float angle = edge.getAngle();

        float line_length = edge.getLength() * 0.8f;
        cv::Point2f direction(cos(angle) * line_length, sin(angle) * line_length);
        cv::Point start_point(static_cast<int>(center.x - direction.x * 0.5f),
            static_cast<int>(center.y - direction.y * 0.5f));
        cv::Point end_point(static_cast<int>(center.x + direction.x * 0.5f),
            static_cast<int>(center.y + direction.y * 0.5f));

        cv::arrowedLine(image, start_point, end_point,
            cv::Scalar(0, 255, 255), 2, cv::LINE_AA, 0, 0.3);

        std::string angle_text = std::to_string(static_cast<int>(angle * 180 / CV_PI));
        cv::putText(image, angle_text,
            cv::Point(static_cast<int>(center.x + 10), static_cast<int>(center.y - 10)),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    }

    //std::cout << "  Angle: " << static_cast<int>(edge.getAngle() * 180 / CV_PI) << "°" << std::endl;
}

void ImageDisplay::visualizeAllChainCodes(const std::vector<EBPTns::Edge>& edges,
    const cv::Mat& image,
    const std::string& filename) {
    cv::Mat viz_image = image.clone();

    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255)
    };

    std::cout << "\n=== Chain code vizualization ===" << std::endl
        << "Edges count: " << edges.size() << std::endl;

    //  тонкие серые линии
    //for (Edge edge : edges) {
    //    const auto& points = edge.getPoints();
    //    for (size_t i = 1; i < points.size(); ++i) {
    //        cv::line(viz_image, points[i - 1], points[i], cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
    //    }
    //}

    //цветные стрелки и направления
    for (size_t i = 0; i < edges.size(); ++i) {
        cv::Scalar color = colors[i % colors.size()];
        visualizeChainCode(edges[i], viz_image, color, true);
    }

    cv::imwrite(filename, viz_image);
}


void ImageDisplay::visualizeAnglesOnly(const std::vector<EBPTns::Edge>& edges,
    const cv::Mat& background,
    const std::string& filename) {
    cv::Mat angle_viz = cv::Mat::zeros(background.size(), CV_8UC3);
    angle_viz.setTo(cv::Scalar(30, 30, 30));

    for (const auto& edge : edges) {
        cv::Point2f center = edge.getCenter();
        float angle = edge.getAngle();
        float length = edge.getLength() * 0.5f;

        cv::Point2f dir(cos(angle) * length, sin(angle) * length);
        cv::Point start(static_cast<int>(center.x - dir.x),
            static_cast<int>(center.y - dir.y));
        cv::Point end(static_cast<int>(center.x + dir.x),
            static_cast<int>(center.y + dir.y));

        int hue = static_cast<int>(angle * 180 / CV_PI) % 180;
        cv::Scalar color;
        if (hue < 60) color = cv::Scalar(0, 255, 0);       // Зеленый
        else if (hue < 120) color = cv::Scalar(255, 0, 0); // Синий
        else color = cv::Scalar(0, 0, 255);                // Красный

        cv::arrowedLine(angle_viz, start, end, color, 2, cv::LINE_AA, 0, 0.3);
        cv::circle(angle_viz, cv::Point(static_cast<int>(center.x),
            static_cast<int>(center.y)),
            3, cv::Scalar(255, 255, 255), -1);
    }

    cv::imwrite(filename, angle_viz);
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
        const auto& group = placed.group;

        // Генерируем цвет для группы
        cv::Scalar color = cv::Scalar(color_dist(rng), color_dist(rng), color_dist(rng));
        cv::Point center(static_cast<int>(group.getCenter().x),
            static_cast<int>(group.getCenter().y));

        // Рисуем центр группы
        cv::circle(placement_map, center, 8, color, -1);
        cv::circle(placement_map, center, 3, cv::Scalar(255, 255, 255), -1);

        // Рисуем выпуклую оболочку из сохраненного hull
        if (!placed.hull.empty() && placed.hull.size() >= 3) {
            for (size_t j = 0; j < placed.hull.size(); ++j) {
                cv::line(placement_map,
                    placed.hull[j],
                    placed.hull[(j + 1) % placed.hull.size()],
                    color, 2);
            }

            // Полупрозрачная заливка
            cv::Mat overlay = placement_map.clone();
            std::vector<std::vector<cv::Point>> hull_contour = { placed.hull };
            cv::fillPoly(overlay, hull_contour, cv::Scalar(color[0], color[1], color[2]));
            cv::addWeighted(overlay, 0.2, placement_map, 0.8, 0, placement_map);

            // Используем bounding box от hull для текста
            cv::Rect bbox = cv::boundingRect(placed.hull);
            std::string edge_count = "E:" + std::to_string(group.getEdges().size()) + " G:" + std::to_string(placed.source_index);
            cv::putText(placement_map, edge_count,
                cv::Point(bbox.x, bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1);
        }
        else {
            cv::Rect bbox = group.getBoundingBox();
            cv::rectangle(placement_map, bbox, color, 2);
            std::string edge_count = "E:" + std::to_string(group.getEdges().size());
            cv::putText(placement_map, edge_count,
                cv::Point(bbox.x, bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1);
        }

        // Рисуем направление (средний угол группы)
        float angle = group.getAverageAngle();
        int line_length = 30;
        cv::Point direction(
            center.x + static_cast<int>(line_length * std::cos(angle)),
            center.y + static_cast<int>(line_length * std::sin(angle))
        );
        cv::line(placement_map, center, direction, color, 2);

        drawArrow(placement_map, direction, angle, color);

        // Добавляем номер группы
        std::string label = std::to_string(i + 1);
        cv::putText(placement_map, label,
            cv::Point(center.x + 10, center.y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(255, 255, 255), 2);
    }

    std::string title = "Placement Map: " + std::to_string(placed_groups.size()) + " groups";
    cv::putText(placement_map, title,
        cv::Point(10, 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2);

    std::string size_info = "Size: " + std::to_string(size.width) + "x" + std::to_string(size.height);
    cv::putText(placement_map, size_info,
        cv::Point(10, 55),
        cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(200, 200, 200), 1);

    return placement_map;
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

void ImageDisplay::visualizeEdgeBins(const cv::Mat& input_image,
    const std::vector<EBPTns::Edge>& edges,
    const std::string& filename) {

    if (edges.empty()) {
        std::cout << "No edges to visualize" << std::endl;
        return;
    }

    // Находим минимальную и максимальную длину для нормализации
    float min_length = edges[0].getLength();
    float max_length = edges[0].getLength();
    for (const auto& edge : edges) {
        float len = edge.getLength();
        if (len < min_length) min_length = len;
        if (len > max_length) max_length = len;
    }

    int img_width = input_image.cols;
    int img_height = input_image.rows;

    // Добавляем отступы между изображениями
    int padding = 20;
    int header_height = 40;

    // Создаем 9 изображений для бинов (каждое такого же размера, как исходное)
    std::vector<cv::Mat> bin_images(9);
    for (int i = 0; i < 9; ++i) {
        if (input_image.channels() == 1) {
            cv::cvtColor(input_image, bin_images[i], cv::COLOR_GRAY2BGR);
        }
        else {
            bin_images[i] = input_image.clone();
        }
        // Затемняем фон, чтобы ребра были видны ярче
        cv::addWeighted(bin_images[i], 0.3, cv::Mat::zeros(bin_images[i].size(), bin_images[i].type()), 0.7, 0, bin_images[i]);
    }

    // Распределяем ребра по бинам
    for (const auto& edge : edges) {
        float length = edge.getLength();
        float angle = edge.getAngle();  // 0 to PI

        // Нормализуем длину к [0, 2] (3 бина)
        float norm_length = 0.0f;
        if (max_length > min_length) {
            norm_length = (length - min_length) / (max_length - min_length) * 2.0f;
        }

        // Определяем бины
        int length_bin = static_cast<int>(norm_length);
        if (length_bin > 2) length_bin = 2;

        int angle_bin = 0;
        if (angle < CV_PI / 3) angle_bin = 0;           // 0-60°
        else if (angle < 2 * CV_PI / 3) angle_bin = 1;  // 60-120°
        else angle_bin = 2;                              // 120-180°

        int bin_index = length_bin * 3 + angle_bin;

        // Рисуем ребро ТОЛЬКО в соответствующем бине
        cv::Mat& bin_img = bin_images[bin_index];
        const auto& points = edge.getPoints();

        // Рисуем линии
        for (size_t i = 1; i < points.size(); ++i) {
            cv::line(bin_img, points[i - 1], points[i],
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        // Рисуем центр ребра
        cv::Point center(static_cast<int>(edge.getCenter().x),
            static_cast<int>(edge.getCenter().y));
        cv::circle(bin_img, center, 3, cv::Scalar(0, 0, 255), -1);
    }

    // Создаем итоговое изображение для отображения всех 9 бинов
    int grid_width = img_width * 3 + padding * 4;
    int grid_height = img_height * 3 + header_height + padding * 4;
    cv::Mat grid = cv::Mat::zeros(grid_height, grid_width, CV_8UC3);
    grid.setTo(cv::Scalar(50, 50, 50));

    // Заголовки
    std::vector<std::string> length_labels = { "Short", "Medium", "Long" };
    std::vector<std::string> angle_labels = { "0-60°", "60-120°", "120-180°" };

    // Добавляем заголовки для углов (сверху)
    for (int col = 0; col < 3; ++col) {
        int x = padding + col * (img_width + padding) + img_width / 2 - 30;
        cv::putText(grid, angle_labels[col],
            cv::Point(x, header_height / 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }

    // Добавляем заголовки для длины (слева)
    for (int row = 0; row < 3; ++row) {
        int y = header_height + padding + row * (img_height + padding) + img_height / 2;
        cv::putText(grid, length_labels[row],
            cv::Point(10, y),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }

    // Размещаем бины на сетке
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            int bin_index = row * 3 + col;

            int x = padding + col * (img_width + padding);
            int y = header_height + padding + row * (img_height + padding);

            // Проверяем, что ROI не выходит за границы
            if (x + img_width <= grid.cols && y + img_height <= grid.rows) {
                cv::Rect roi(x, y, img_width, img_height);
                bin_images[bin_index].copyTo(grid(roi));

                // Рамка вокруг бина
                cv::rectangle(grid, roi, cv::Scalar(200, 200, 200), 2);

                // Количество ребер в бине
                int edge_count = 0;
                for (const auto& edge : edges) {
                    float angle = edge.getAngle();
                    float length = edge.getLength();
                    float norm_length = (length - min_length) / (max_length - min_length) * 2.0f;
                    int l_bin = static_cast<int>(norm_length);
                    if (l_bin > 2) l_bin = 2;

                    int a_bin = 0;
                    if (angle < CV_PI / 3) a_bin = 0;
                    else if (angle < 2 * CV_PI / 3) a_bin = 1;
                    else a_bin = 2;

                    if (l_bin == row && a_bin == col) edge_count++;
                }

                std::string count_text = "n=" + std::to_string(edge_count);
                cv::putText(grid, count_text,
                    cv::Point(x + img_width - 80, y + img_height - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            }
        }
    }

    cv::imwrite(filename, grid);
    std::cout << "Bins visualization saved: " << filename << std::endl;
}

void ImageDisplay::visualizeBinDistribution(const std::vector<EBPTns::Edge>& edges,
    const std::string& filename) {

    if (edges.empty()) return;

    float min_len = edges[0].getLength();
    float max_len = edges[0].getLength();
    for (const auto& e : edges) {
        min_len = std::min(min_len, e.getLength());
        max_len = std::max(max_len, e.getLength());
    }

    int bins[3][3] = { {0} };

    for (const auto& edge : edges) {
        float len = edge.getLength();
        float angle = edge.getAngle();

        int len_bin;
        if (max_len - min_len < 0.1f) len_bin = 1;
        else {
            float norm = (len - min_len) / (max_len - min_len);
            if (norm < 0.33f) len_bin = 0;
            else if (norm < 0.66f) len_bin = 1;
            else len_bin = 2;
        }

        int angle_bin;
        if (angle < CV_PI / 3) angle_bin = 0;
        else if (angle < 2 * CV_PI / 3) angle_bin = 1;
        else angle_bin = 2;

        bins[len_bin][angle_bin]++;
    }

    int cell_size = 100;
    cv::Mat dist_viz = cv::Mat::zeros(cell_size * 4, cell_size * 4, CV_8UC3);
    dist_viz.setTo(cv::Scalar(240, 240, 240));

    std::vector<std::string> len_labels = { "Short", "Medium", "Long" };
    std::vector<std::string> angle_labels = { "0-60°", "60-120°", "120-180°" };

    // Рисуем таблицу
    for (int i = 0; i <= 3; ++i) {
        cv::line(dist_viz, cv::Point(i * cell_size, 0),
            cv::Point(i * cell_size, cell_size * 3),
            cv::Scalar(0, 0, 0), 2);
        cv::line(dist_viz, cv::Point(0, i * cell_size),
            cv::Point(cell_size * 3, i * cell_size),
            cv::Scalar(0, 0, 0), 2);
    }

    // Заполняем ячейки
    int total = edges.size();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            int count = bins[row][col];
            float percent = 100.0f * count / total;

            cv::Rect cell(col * cell_size, row * cell_size, cell_size, cell_size);

            int intensity = static_cast<int>(120 + 100 * percent / 50);
            intensity = std::min(255, intensity);
            cv::rectangle(dist_viz, cell, cv::Scalar(255 - intensity, 255, 255 - intensity), -1);

            std::string text = std::to_string(count);
            cv::putText(dist_viz, text,
                cv::Point(col * cell_size + 20, row * cell_size + 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

            std::string percent_text = std::to_string(static_cast<int>(percent)) + "%";
            cv::putText(dist_viz, percent_text,
                cv::Point(col * cell_size + 20, row * cell_size + 70),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 80, 80), 1);
        }
    }

    for (int i = 0; i < 3; ++i) {
        cv::putText(dist_viz, angle_labels[i],
            cv::Point(i * cell_size + 20, cell_size * 3 + 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        cv::putText(dist_viz, len_labels[i],
            cv::Point(cell_size * 3 + 10, i * cell_size + 40),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    cv::putText(dist_viz, "Angle →", cv::Point(cell_size, cell_size * 3 + 50),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::putText(dist_viz, "Length ↓", cv::Point(cell_size * 3 + 20, cell_size),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    cv::imwrite(filename, dist_viz);
    std::cout << "Распределение по бинам: " << filename << std::endl;

    std::cout << "\nТаблица распределения ребер по бинам (длина × угол):" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "          0-60°   60-120°  120-180°  Total" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    for (int row = 0; row < 3; ++row) {
        std::cout << len_labels[row] << "   ";
        int row_total = 0;
        for (int col = 0; col < 3; ++col) {
            std::cout << "  " << bins[row][col] << "\t";
            row_total += bins[row][col];
        }
        std::cout << "  " << row_total << std::endl;
    }
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Total: " << total << " ребер" << std::endl;
}