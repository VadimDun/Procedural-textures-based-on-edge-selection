#include "AppController.h"
#include "TextureAnalysis.h"
#include "TextureSynthesis.h"
#include "PixelSynthesis.h"
#include "PlacedGroup.h"
#include "SourceGroupInfo.h"
#include "ImageDisplay.h"
#include "Config.h"

// Qt includes
#include <QThread>
#include <QElapsedTimer>
#include <QDateTime>
#include <QEventLoop>
#include <QTimer>

#include <QImage>
#include <QFile>
#include <QByteArray>

// Standard includes
#include <chrono>
#include <iostream>


// ======================== AnalysisWorker Implementation ========================

AnalysisWorker::AnalysisWorker(AppController* controller, const cv::Mat& image,
    int minEdgeLength, int regionSize, float ruler, double threshold)
    : QObject(nullptr)
    , controller_(controller)
    , image_(image)
    , minEdgeLength_(minEdgeLength)
    , regionSize_(regionSize)
    , ruler_(ruler)
    , threshold_(threshold)
{
}

void AnalysisWorker::doWork() {
    try {
        auto textureAnalysis = controller_->getOrCreateTextureAnalysis(minEdgeLength_, regionSize_, ruler_, threshold_);

        if (!textureAnalysis) {
            emit finished(false, "Failed to create or initialize texture analysis");
            return;
        }

        emit progress(10);

        if (controller_->isCancelled()) {
            emit finished(false, "Analysis cancelled");
            return;
        }

        emit progress(30);

        auto result = textureAnalysis->analyzeTexture(image_, MODEL_PATH);

        emit progress(70);

        if (!result.isValid()) {
            emit finished(false, "Analysis failed: no edges found");
            return;
        }

        emit progress(90);

        // Сохраняем результат в контроллере
        controller_->setAnalysisResult(std::make_unique<EBPTns::AnalysisResult>(std::move(result)));

        // Создаем визуализацию групп
        const auto* analysisResult = controller_->getAnalysisResult();
        if (analysisResult) {
            cv::Mat groupsVis = ImageDisplay::visualizeGroups(image_,
                analysisResult->source_groups);
            emit visualizationReady(groupsVis);
        }

        emit progress(100);
        emit finished(true, "Analysis completed successfully");

    }
    catch (const std::exception& e) {
        emit finished(false, QString("Exception: ") + e.what());
    }
    catch (...) {
        emit finished(false, "Unknown exception occurred");
    }
}

// ======================== SynthesisWorker Implementation ========================

SynthesisWorker::SynthesisWorker(AppController* controller, const cv::Mat& image,
    const EBPTns::AnalysisResult* analysisResult,
	const cv::Size& outputSize, bool enableRotation, bool enableScaling,
    float angleSpread, unsigned int seed,
    float largeFillPercentage, float mediumFillPercentage, float smallFillPercentage)
    : controller_(controller)
    , image_(image)
    , analysisResult_(analysisResult)
    , outputSize_(outputSize)
    , enableRotation_(enableRotation)
	, enableScaling_(enableScaling)
    , angleSpread_(angleSpread)
    , seed_(seed)
    , largeFillPercentage_(largeFillPercentage)
    , mediumFillPercentage_(mediumFillPercentage)
    , smallFillPercentage_(smallFillPercentage)
{
}

void SynthesisWorker::doWork() {
    try {
        if (!analysisResult_ || !analysisResult_->isValid()) {
            emit finished(false, "No valid analysis result");
            return;
        }

        emit progress(10);

        auto textureSynthesis = controller_->getOrCreateTextureSynthesis(
            outputSize_, enableRotation_, enableScaling_, seed_);

        textureSynthesis->setTargetFillPercentage(
            largeFillPercentage_,
            mediumFillPercentage_,
            smallFillPercentage_
        );

        emit progress(30);

        if (controller_->isCancelled()) {
            emit finished(false, "Synthesis cancelled");
            return;
        }

        auto placedGroups = textureSynthesis->synthesizeHierarchicalPlacement(
            image_, analysisResult_->source_groups);

        emit progress(60);

        if (controller_->isCancelled()) {
            emit finished(false, "Synthesis cancelled");
            return;
        }

        // Сохраняем карту размещения
        cv::Mat placementMap = ImageDisplay::drawPlacementMap(
            placedGroups, outputSize_);
        controller_->setPlacementMap(placementMap);

        emit progress(70);
        emit placementReady(placementMap);

        if (controller_->isCancelled()) {
            emit finished(false, "Synthesis cancelled");
            return;
        }

        // Заполнение пикселей
        EBPTns::PixelSynthesis pixelSynthesis;
        cv::Mat outputTexture = pixelSynthesis.fillPixels(
            image_, analysisResult_->source_groups, placedGroups, outputSize_);
        controller_->setOutputTexture(outputTexture);

        emit progress(90);
        emit textureReady(outputTexture);

        emit progress(100);
        emit finished(true, "Synthesis completed successfully");

    }
    catch (const std::exception& e) {
        emit finished(false, QString("Exception: ") + e.what());
    }
    catch (...) {
        emit finished(false, "Unknown exception occurred");
    }
}

// ======================== AppController Implementation ========================

AppController::AppController(QObject* parent)
    : QObject(parent)
    , isCancelled_(false)
    , isAnalyzed_(false)
    , isTextureAnalysisInitialized_(false)
{
    resetState();
    emitLog("AppController initialized");
}

AppController::~AppController() {
    cancel();
}

bool AppController::loadImage(const QString& path) {
    if (path.isEmpty()) {
        emitLog("Failed to load image: empty path");
        return false;
    }

    // Проверяем существование файла через Qt
    QFile file(path);
    if (!file.exists()) {
        emitLog("Failed to load image: file not found - " + path);
        return false;
    }

    // Загружаем через Qt в QImage
    QImage qimg;
    if (!qimg.load(path)) {
        emitLog("Failed to load image: QImage::load failed - " + path);
        return false;
    }

    // Конвертируем QImage в cv::Mat без использования std::string
    QImage image = qimg.convertToFormat(QImage::Format_RGB888);

    if (image.isNull()) {
        emitLog("Failed to load image: format conversion failed - " + path);
        return false;
    }

    // Создаём cv::Mat напрямую из данных QImage
    cv::Mat img(image.height(), image.width(), CV_8UC3);

    // Копируем данные построчно
    for (int y = 0; y < image.height(); ++y) {
        const uchar* srcLine = image.scanLine(y);
        uchar* dstLine = img.ptr<uchar>(y);
        memcpy(dstLine, srcLine, image.width() * 3);
    }

    if (img.empty()) {
        emitLog("Failed to load image: cv::Mat creation failed - " + path);
        return false;
    }

    // Конвертируем RGB в BGR для OpenCV
    cv::cvtColor(img, originalImage_, cv::COLOR_RGB2BGR);

    resetState();
    isAnalyzed_ = false;
    analysisResult_.reset();
    placementMap_.release();
    outputTexture_.release();

    emitLog("Image loaded: " + path +
        QString(" (%1x%2)").arg(originalImage_.cols).arg(originalImage_.rows));
    return true;
}


void AppController::analyze() {
    if (originalImage_.empty()) {
        emit analysisFinished(false, "No image loaded");
        return;
    }

    isCancelled_ = false;
    emit analysisStarted();
    emitLog("Starting texture analysis...");

    AnalysisWorker* worker = new AnalysisWorker(this, originalImage_,
        minEdgeLength_, superpixelRegionSize_, superpixelRuler_, superpixelThreshold_);

    QThread* thread = new QThread();
    worker->moveToThread(thread);

    // Подключаем сигналы
    connect(thread, &QThread::started, worker, &AnalysisWorker::doWork);
    connect(worker, &AnalysisWorker::finished, this, [this, thread, worker](bool success, const QString& msg) {
        if (success && analysisResult_) {
            isAnalyzed_ = true;
        }
        emit analysisFinished(success, msg);
        thread->quit();
        worker->deleteLater();
        thread->deleteLater();
        });
    connect(worker, &AnalysisWorker::progress, this, &AppController::analysisProgress);
    connect(worker, &AnalysisWorker::visualizationReady, this, &AppController::analysisGroupsVisualization);

    // Обработка ошибок потока
    connect(thread, &QThread::finished, thread, &QThread::deleteLater);

    thread->start();
}

void AppController::synthesize() {
    if (!analysisResult_ || !analysisResult_->isValid()) {
        emitLog("No valid analysis result. Please run analysis first.");
        emit synthesisFinished(false, "No analysis result");
        return;
    }

    if (originalImage_.empty()) {
        emitLog("No original image loaded");
        emit synthesisFinished(false, "No image loaded");
        return;
    }

    emit synthesisStarted();

    QThread* thread = new QThread();
    SynthesisWorker* worker = new SynthesisWorker(
        this,
        originalImage_,
        analysisResult_.get(),
        cv::Size(outputWidth_, outputHeight_),
        enableRotation_,
        enableScaling_,
        angleSpread_,
        randomSeed_,
        largeFillPercentage_,
        mediumFillPercentage_,
        smallFillPercentage_
    );

    worker->moveToThread(thread);

    // Подключаем сигналы
    connect(thread, &QThread::started, worker, &SynthesisWorker::doWork);
    connect(worker, &SynthesisWorker::finished, thread, &QThread::quit);
    connect(worker, &SynthesisWorker::finished, worker, &SynthesisWorker::deleteLater);
    connect(thread, &QThread::finished, thread, &QThread::deleteLater);

    // Проксируем сигналы
    connect(worker, &SynthesisWorker::progress, this, &AppController::synthesisProgress);
    connect(worker, &SynthesisWorker::finished, this, &AppController::synthesisFinished);
    connect(worker, &SynthesisWorker::placementReady, this, &AppController::synthesisPlacementUpdated);
    connect(worker, &SynthesisWorker::textureReady, this, &AppController::synthesisTextureUpdated);

    thread->start();
}

void AppController::cancel() {
    isCancelled_ = true;
    emitLog("Operation cancelled by user");
}

void AppController::resetState() {
    // Не сбрасываем originalImage_, только результаты
    isAnalyzed_ = false;
    // Не удаляем analysisResult_ здесь, чтобы не повредить возможность регенерации
}

void AppController::emitLog(const QString& msg) {
    emit logMessage(QDateTime::currentDateTime().toString("hh:mm:ss") + " - " + msg);
}

void AppController::setAnalysisResult(std::unique_ptr<EBPTns::AnalysisResult> result) {
    analysisResult_ = std::move(result);
}

const EBPTns::AnalysisResult* AppController::getAnalysisResult() const {
    return analysisResult_.get();
}

void AppController::setPlacementMap(const cv::Mat& map) {
    placementMap_ = map;
}

void AppController::setOutputTexture(const cv::Mat& texture) {
    outputTexture_ = texture;
}

std::shared_ptr<EBPTns::TextureSynthesis> AppController::getOrCreateTextureSynthesis(
    const cv::Size& outputSize, bool enableRotation, bool enableScaling, unsigned int seed) {

    // Проверяем, нужно ли создать новый синтезатор
    bool needsNew = !textureSynthesis_ ||
        textureSynthesis_->getOutputSize() != outputSize ||
        textureSynthesis_->isRotationEnabled() != enableRotation ||
        textureSynthesis_->isScalingEnabled() != enableScaling ||
        textureSynthesis_->getRandomSeed() != seed;

    if (needsNew) {
        textureSynthesis_ = std::make_shared<EBPTns::TextureSynthesis>(outputSize, enableRotation, enableScaling);
        textureSynthesis_->setRandomSeed(seed);
        std::cout << "===========================created new textureSynthesis_" << std::endl;
    }
    else {
        std::cout << "===========================reusing existing textureSynthesis_" << std::endl;
    }

    return textureSynthesis_;
}

std::shared_ptr<EBPTns::TextureAnalysis> AppController::getOrCreateTextureAnalysis(
    int minEdgeLength, int regionSize, float ruler, double threshold) {

    bool needsNew = !textureAnalysis_;

    if (!needsNew) {
        // Проверяем, изменились ли параметры
        needsNew = (minEdgeLength_ != textureAnalysis_->getMinEdgeLength()) ||
            (superpixelRegionSize_ != textureAnalysis_->getSuperpixelRegionSize()) ||
            (superpixelRuler_ != textureAnalysis_->getSuperpixelRuler()) ||
            (superpixelThreshold_ != textureAnalysis_->getSuperpixelThreshold());
    }
    else {
        textureAnalysis_ = std::make_shared<EBPTns::TextureAnalysis>();
        textureAnalysis_->initializeStructuredDetector(MODEL_PATH);
    }

    if (needsNew) {
        textureAnalysis_->setMinEdgeLength(minEdgeLength);
        textureAnalysis_->setSuperpixelParams(regionSize, ruler, threshold);

        // Обновляем сохраненные параметры
        minEdgeLength_ = minEdgeLength;
        superpixelRegionSize_ = regionSize;
        superpixelRuler_ = ruler;
        superpixelThreshold_ = threshold;

        std::cout << "===========================created new textureAnalysis_ with params: "
            << "minEdgeLength=" << minEdgeLength
            << ", regionSize=" << regionSize
            << ", ruler=" << ruler
            << ", threshold=" << threshold << std::endl;
    }
    else {
        std::cout << "===========================reusing existing textureAnalysis_" << std::endl;
    }

    return textureAnalysis_;
}
