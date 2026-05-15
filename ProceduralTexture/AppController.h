#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <QObject>
#include <QString>
#include <QImage>

// Forward declarations
namespace EBPTns {
    struct SourceGroupInfo;
    struct PlacedGroup;
    class TextureAnalysis;
    class TextureSynthesis;
    struct AnalysisResult;
}
class AppController;

// ======================== Worker Declaration ========================

class AnalysisWorker : public QObject {
    Q_OBJECT
public:
    AnalysisWorker(AppController* controller, const cv::Mat& image,
        int minEdgeLength, int regionSize, double threshold);

public slots:
    void doWork();

signals:
    void finished(bool success, const QString& message);
    void progress(int percent);
    void visualizationReady(const cv::Mat& vis);

private:
    AppController* controller_;
    cv::Mat image_;
    int minEdgeLength_;
    int regionSize_;
    double threshold_;
};

class SynthesisWorker : public QObject {
    Q_OBJECT
public:
    SynthesisWorker(AppController* controller, const cv::Mat& image,
        const EBPTns::AnalysisResult* analysisResult,
        const cv::Size& outputSize, bool enableRotation,
        float angleSpread, unsigned int seed,
        float largeFillPercentage, float mediumFillPercentage, float smallFillPercentage);

public slots:
    void doWork();

signals:
    void finished(bool success, const QString& message);
    void progress(int percent);
    void placementReady(const cv::Mat& placement);
    void textureReady(const cv::Mat& texture);

private:
    AppController* controller_;
    cv::Mat image_;
    const EBPTns::AnalysisResult* analysisResult_;
    cv::Size outputSize_;
    bool enableRotation_;
    float angleSpread_;
    unsigned int seed_;
    float largeFillPercentage_;
    float mediumFillPercentage_;
    float smallFillPercentage_;
};

// ======================== AppController Declaration ========================

class AppController : public QObject {
    Q_OBJECT

public:
    explicit AppController(QObject* parent = nullptr);
    ~AppController();

    // Загрузка и состояние
    bool loadImage(const QString& path);
    bool hasImage() const { return !originalImage_.empty(); }
    cv::Mat getOriginalImage() const { return originalImage_; }
    bool isCancelled() const { return isCancelled_; }

    // Параметры анализа
    void setMinEdgeLength(int value) { minEdgeLength_ = value; }
    void setSuperpixelRegionSize(int value) { superpixelRegionSize_ = value; }
    void setSuperpixelThreshold(double value) { superpixelThreshold_ = value; }

    // Параметры синтеза
    void setOutputSize(int width, int height) { outputWidth_ = width; outputHeight_ = height; }
    void setEnableRotation(bool enable) { enableRotation_ = enable; }
    void setAngleSpread(float spread) { angleSpread_ = spread; }
    void setRandomSeed(unsigned int seed) { randomSeed_ = seed; }

    void setLargeFillPercentage(int percent) { largeFillPercentage_ = percent / 100.0f; }
    void setMediumFillPercentage(int percent) { mediumFillPercentage_ = percent / 100.0f; }
    void setSmallFillPercentage(int percent) { smallFillPercentage_ = percent / 100.0f; }

    int getLargeFillPercentageInt() const { return static_cast<int>(largeFillPercentage_ * 100); }
    int getMediumFillPercentageInt() const { return static_cast<int>(mediumFillPercentage_ * 100); }
    int getSmallFillPercentageInt() const { return static_cast<int>(smallFillPercentage_ * 100); }

    // Действия
    void analyze();
    void synthesize();
    void cancel();

    // Получение результатов
    cv::Mat getPlacementMap() const { return placementMap_; }
    cv::Mat getOutputTexture() const { return outputTexture_; }
    void setAnalysisResult(std::unique_ptr<EBPTns::AnalysisResult> result);
    const EBPTns::AnalysisResult* getAnalysisResult() const;
    void setPlacementMap(const cv::Mat& map);
    void setOutputTexture(const cv::Mat& texture);

    std::shared_ptr<EBPTns::TextureSynthesis> getOrCreateTextureSynthesis(
        const cv::Size& outputSize, bool enableRotation, unsigned int seed);

signals:
    void analysisStarted();
    void analysisProgress(int percent);
    void analysisFinished(bool success, const QString& message);
    void analysisGroupsVisualization(const cv::Mat& visualization);

    void synthesisStarted();
    void synthesisProgress(int percent);
    void synthesisFinished(bool success, const QString& message);
    void synthesisPlacementUpdated(const cv::Mat& placementMap);
    void synthesisTextureUpdated(const cv::Mat& texture);

    void logMessage(const QString& message);

private:
    void resetState();
    void emitLog(const QString& msg);

    // Данные
    cv::Mat originalImage_;
    cv::Mat placementMap_;
    cv::Mat outputTexture_;

    // Результаты анализа
    std::unique_ptr<EBPTns::AnalysisResult> analysisResult_;

    std::shared_ptr<EBPTns::TextureSynthesis> textureSynthesis_;

    // Параметры
    int minEdgeLength_ = 15;
    int superpixelRegionSize_ = 120;
    double superpixelThreshold_ = 0.1;

    int outputWidth_ = 1400;
    int outputHeight_ = 1000;
    bool enableRotation_ = true;
    float angleSpread_ = 0.1f;
    unsigned int randomSeed_ = 42;

    float largeFillPercentage_ = 0.5f;
    float mediumFillPercentage_ = 0.85f;
    float smallFillPercentage_ = 0.98f;

    // Состояние
    bool isCancelled_ = false;
    bool isAnalyzed_ = false;

    friend class AnalysisWorker;
    friend class SynthesisWorker;
};
