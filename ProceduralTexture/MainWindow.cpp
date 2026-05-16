#include "MainWindow.h"
#include "AppController.h"

// Qt Core
#include <QThread>
#include <QDateTime>
#include <QTimer>

// Qt GUI
#include <QCloseEvent>

// Qt Widgets
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QTextEdit>
#include <QProgressBar>
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QTabWidget>
#include <QScrollArea>
#include <QSplitter>
#include <QDockWidget>
#include <QTextCursor>

// OpenCV
#include <opencv2/opencv.hpp>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , controller_(std::make_unique<AppController>(this))
{
    setupUi();

    // Подключаем сигналы контроллера
    connect(controller_.get(), &AppController::analysisFinished,
        this, &MainWindow::onAnalysisFinished);
    connect(controller_.get(), &AppController::synthesisFinished,
        this, &MainWindow::onSynthesisFinished);
    connect(controller_.get(), &AppController::logMessage,
        this, &MainWindow::onLogMessage);
    connect(controller_.get(), &AppController::analysisProgress,
        this, &MainWindow::onAnalysisProgress);
    connect(controller_.get(), &AppController::synthesisProgress,
        this, &MainWindow::onSynthesisProgress);
    connect(controller_.get(), &AppController::analysisGroupsVisualization,
        this, &MainWindow::onAnalysisGroupsVisualization);
    connect(controller_.get(), &AppController::synthesisPlacementUpdated,
        this, &MainWindow::onSynthesisPlacementUpdated);
    connect(controller_.get(), &AppController::synthesisTextureUpdated,
        this, &MainWindow::onSynthesisTextureUpdated);

    setWindowTitle("Texture Synthesis Application");
    resize(1400, 900);

    updateButtonStates();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUi() {
    createMenuBar();
    createToolBar();
    createCentralWidget();
    createLogPanel();

    // Статус бар
    statusLabel_ = new QLabel("Ready");
    statusBar()->addWidget(statusLabel_);

    progressBar_ = new QProgressBar();
    progressBar_->setMaximumWidth(200);
    progressBar_->setMinimumWidth(100);
    progressBar_->setVisible(false);
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    progressBar_->setTextVisible(true);
    statusBar()->addPermanentWidget(progressBar_);
}

void MainWindow::createMenuBar() {
    QMenu* fileMenu = menuBar()->addMenu("&File");

    QAction* loadAction = fileMenu->addAction("&Load Image");
    loadAction->setShortcut(QKeySequence::Open);
    connect(loadAction, &QAction::triggered, this, &MainWindow::onLoadImage);

    QAction* saveResultAction = fileMenu->addAction("&Save Result");
    saveResultAction->setShortcut(QKeySequence::Save);
    connect(saveResultAction, &QAction::triggered, this, &MainWindow::onSaveResult);

    QAction* savePlacementAction = fileMenu->addAction("Save &Placement Map");
    connect(savePlacementAction, &QAction::triggered, this, &MainWindow::onSavePlacement);

    fileMenu->addSeparator();

    QAction* exitAction = fileMenu->addAction("E&xit");
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);

    QMenu* helpMenu = menuBar()->addMenu("&Help");
    QAction* aboutAction = helpMenu->addAction("&About");
    connect(aboutAction, &QAction::triggered, [this]() {
        QMessageBox::about(this, "About Texture Synthesis",
            "Texture Synthesis Application\n"
            "Based on Edge-Based Patch Texture Synthesis");
        });
}

void MainWindow::createToolBar() {
    QToolBar* toolBar = addToolBar("Main");
    toolBar->setMovable(false);

    loadImageButton_ = new QPushButton("Load Image");
    connect(loadImageButton_, &QPushButton::clicked, this, &MainWindow::onLoadImage);
    toolBar->addWidget(loadImageButton_);

    toolBar->addSeparator();

    analyzeButton_ = new QPushButton("Analyze");
    analyzeButton_->setEnabled(false);
    connect(analyzeButton_, &QPushButton::clicked, this, &MainWindow::onAnalyze);
    toolBar->addWidget(analyzeButton_);

    synthesizeButton_ = new QPushButton("Synthesize");
    synthesizeButton_->setEnabled(false);
    connect(synthesizeButton_, &QPushButton::clicked, this, &MainWindow::onSynthesize);
    toolBar->addWidget(synthesizeButton_);

    toolBar->addSeparator();

    cancelButton_ = new QPushButton("Cancel");
    cancelButton_->setEnabled(false);
    connect(cancelButton_, &QPushButton::clicked, this, &MainWindow::onCancel);
    toolBar->addWidget(cancelButton_);

    toolBar->addSeparator();

    saveResultButton_ = new QPushButton("Save Result");
    saveResultButton_->setEnabled(false);
    connect(saveResultButton_, &QPushButton::clicked, this, &MainWindow::onSaveResult);
    toolBar->addWidget(saveResultButton_);

    savePlacementButton_ = new QPushButton("Save Placement");
    savePlacementButton_->setEnabled(false);
    connect(savePlacementButton_, &QPushButton::clicked, this, &MainWindow::onSavePlacement);
    toolBar->addWidget(savePlacementButton_);
}

void MainWindow::createCentralWidget() {
    QWidget* centralWidget = new QWidget(this);
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);

    // Левая панель - параметры
    QWidget* leftPanel = new QWidget();
    QVBoxLayout* leftLayout = new QVBoxLayout(leftPanel);

    createParametersPanel();
    leftLayout->addWidget(analysisParamsGroup_);
    leftLayout->addWidget(synthesisParamsGroup_);
    leftLayout->addStretch();

    // Центральная область - изображения
    createImageWidgets();

    // Создаем сплиттер между параметрами и изображениями
    QSplitter* splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(leftPanel);
    splitter->addWidget(imageTabWidget_);
    splitter->setStretchFactor(0, 0);  // Левая панель не растягивается
    splitter->setStretchFactor(1, 1);  // Изображения растягиваются

    mainLayout->addWidget(splitter);

    setCentralWidget(centralWidget);
}

void MainWindow::createParametersPanel() {
    // === Параметры анализа ===
    analysisParamsGroup_ = new QGroupBox("Analysis Parameters");
    QGridLayout* analysisLayout = new QGridLayout(analysisParamsGroup_);

    analysisLayout->addWidget(new QLabel("Min Edge Length:"), 0, 0);
    minEdgeLengthSpin_ = new QSpinBox();
    minEdgeLengthSpin_->setRange(5, 100);
    minEdgeLengthSpin_->setValue(10);
    minEdgeLengthSpin_->setSuffix(" px");
    connect(minEdgeLengthSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
        [this](int v) { controller_->setMinEdgeLength(v); });
    analysisLayout->addWidget(minEdgeLengthSpin_, 0, 1);

    analysisLayout->addWidget(new QLabel("Superpixel Size:"), 1, 0);
    superpixelSizeSpin_ = new QSpinBox();
    superpixelSizeSpin_->setRange(30, 300);
    superpixelSizeSpin_->setValue(120);
    superpixelSizeSpin_->setSuffix(" px");
    connect(superpixelSizeSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
        [this](int v) { controller_->setSuperpixelRegionSize(v); });
    analysisLayout->addWidget(superpixelSizeSpin_, 1, 1);

    analysisLayout->addWidget(new QLabel("Edge Threshold:"), 2, 0);
    thresholdSpin_ = new QDoubleSpinBox();
    thresholdSpin_->setRange(0.01, 0.5);
    thresholdSpin_->setSingleStep(0.05);
    thresholdSpin_->setValue(0.1);
    connect(thresholdSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double v) { controller_->setSuperpixelThreshold(v); });
    analysisLayout->addWidget(thresholdSpin_, 2, 1);

    analysisLayout->addWidget(new QLabel("Superpixel Ruler:"), 3, 0);
    rulerSpin_ = new QDoubleSpinBox();
    rulerSpin_->setRange(5, 30);
    rulerSpin_->setValue(10);
    rulerSpin_->setSingleStep(1);
    analysisLayout->addWidget(rulerSpin_, 3, 1);

    // === Параметры синтеза ===
    synthesisParamsGroup_ = new QGroupBox("Synthesis Parameters");
    QGridLayout* synthesisLayout = new QGridLayout(synthesisParamsGroup_);

    synthesisLayout->addWidget(new QLabel("Output Width:"), 0, 0);
    outputWidthSpin_ = new QSpinBox();
    outputWidthSpin_->setRange(100, 4000);
    outputWidthSpin_->setValue(1200);
    outputWidthSpin_->setSuffix(" px");
    outputWidthSpin_->setSingleStep(100);
    connect(outputWidthSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
        [this](int w) { controller_->setOutputSize(w, outputHeightSpin_->value()); });
    synthesisLayout->addWidget(outputWidthSpin_, 0, 1);

    synthesisLayout->addWidget(new QLabel("Output Height:"), 1, 0);
    outputHeightSpin_ = new QSpinBox();
    outputHeightSpin_->setRange(100, 4000);
    outputHeightSpin_->setValue(800);
    outputHeightSpin_->setSuffix(" px");
    outputHeightSpin_->setSingleStep(100);
    connect(outputHeightSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
        [this](int h) { controller_->setOutputSize(outputWidthSpin_->value(), h); });
    synthesisLayout->addWidget(outputHeightSpin_, 1, 1);

    synthesisLayout->addWidget(new QLabel("Enable Rotation:"), 2, 0);
    rotationCheckBox_ = new QCheckBox();
    rotationCheckBox_->setChecked(true);
    connect(rotationCheckBox_, &QCheckBox::toggled,
        [this](bool checked) {
            controller_->setEnableRotation(checked);
            updateParameterVisibility();
        });
    synthesisLayout->addWidget(rotationCheckBox_, 2, 1);

    synthesisLayout->addWidget(new QLabel("Enable Scaling:"), 3, 0);
    scalingCheckBox_ = new QCheckBox();
    scalingCheckBox_->setChecked(true);
    connect(scalingCheckBox_, &QCheckBox::toggled,
        [this](bool checked) {
            controller_->setEnableScaling(checked);
            updateParameterVisibility();
        });
    synthesisLayout->addWidget(scalingCheckBox_, 3, 1);

    synthesisLayout->addWidget(new QLabel("Angle Spread:"), 4, 0);
    angleSpreadSpin_ = new QDoubleSpinBox();
    angleSpreadSpin_->setRange(0.0, 0.5);
    angleSpreadSpin_->setSingleStep(0.05);
    angleSpreadSpin_->setValue(0.1);
    angleSpreadSpin_->setDecimals(2);
    connect(angleSpreadSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double v) { controller_->setAngleSpread(static_cast<float>(v)); });
    synthesisLayout->addWidget(angleSpreadSpin_, 4, 1);

    synthesisLayout->addWidget(new QLabel("Random Seed:"), 5, 0);
    randomSeedSpin_ = new QSpinBox();
    randomSeedSpin_->setRange(0, 999999);
    randomSeedSpin_->setValue(42);
    connect(randomSeedSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
        [this](int v) { controller_->setRandomSeed(static_cast<unsigned int>(v)); });
    synthesisLayout->addWidget(randomSeedSpin_, 5, 1);

    // Добавляем разделитель
    QFrame* line = new QFrame();
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    synthesisLayout->addWidget(line, 6, 0, 1, 2);

    synthesisLayout->addWidget(new QLabel("Large Elements Fill %:"), 7, 0);
    largeFillSpin_ = new QSpinBox();
    largeFillSpin_->setRange(0, 100);
    largeFillSpin_->setValue(50);
    largeFillSpin_->setSuffix(" %");
    largeFillSpin_->setToolTip("Target fill percentage for large elements (0-100%)");
    connect(largeFillSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &MainWindow::onLargeFillChanged);
    synthesisLayout->addWidget(largeFillSpin_, 7, 1);

    synthesisLayout->addWidget(new QLabel("Medium Elements Fill %:"), 8, 0);
    mediumFillSpin_ = new QSpinBox();
    mediumFillSpin_->setRange(0, 100);
    mediumFillSpin_->setValue(85);
    mediumFillSpin_->setSuffix(" %");
    mediumFillSpin_->setToolTip("Target fill percentage for medium elements (0-100%)");
    connect(mediumFillSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &MainWindow::onMediumFillChanged);
    synthesisLayout->addWidget(mediumFillSpin_, 8, 1);

    synthesisLayout->addWidget(new QLabel("Small Elements Fill %:"), 9, 0);
    smallFillSpin_ = new QSpinBox();
    smallFillSpin_->setRange(0, 100);
    smallFillSpin_->setValue(98);
    smallFillSpin_->setSuffix(" %");
    smallFillSpin_->setToolTip("Target fill percentage for small elements (0-100%)");
    connect(smallFillSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &MainWindow::onSmallFillChanged);
    synthesisLayout->addWidget(smallFillSpin_, 9, 1);

    updateParameterVisibility();
}

void MainWindow::onLargeFillChanged(int value) {
    controller_->setLargeFillPercentage(value);
    onLogMessage(QString("Large fill percentage changed to: %1%").arg(value));
}

void MainWindow::onMediumFillChanged(int value) {
    controller_->setMediumFillPercentage(value);
    onLogMessage(QString("Medium fill percentage changed to: %1%").arg(value));
}

void MainWindow::onSmallFillChanged(int value) {
    controller_->setSmallFillPercentage(value);
    onLogMessage(QString("Small fill percentage changed to: %1%").arg(value));
}

void MainWindow::updateParameterVisibility() {
    angleSpreadSpin_->setEnabled(rotationCheckBox_->isChecked());
}

void MainWindow::createImageWidgets() {
    imageTabWidget_ = new QTabWidget();

    // Вкладка с исходным изображением
    QWidget* originalTab = new QWidget();
    QVBoxLayout* originalLayout = new QVBoxLayout(originalTab);
    originalImageLabel_ = new QLabel();
    originalImageLabel_->setAlignment(Qt::AlignCenter);
    originalImageLabel_->setStyleSheet("border: 1px solid gray; background-color: #2a2a2a;");
    originalImageLabel_->setMinimumSize(400, 400);
    originalImageLabel_->setText("No image loaded\n\nClick 'Load Image' to start");
    QScrollArea* originalScroll = new QScrollArea();
    originalScroll->setWidget(originalImageLabel_);
    originalScroll->setWidgetResizable(true);
    originalLayout->addWidget(originalScroll);
    imageTabWidget_->addTab(originalTab, "Original Image");

    // Вкладка с группами (результат анализа)
    QWidget* groupsTab = new QWidget();
    QVBoxLayout* groupsLayout = new QVBoxLayout(groupsTab);
    groupsImageLabel_ = new QLabel();
    groupsImageLabel_->setAlignment(Qt::AlignCenter);
    groupsImageLabel_->setStyleSheet("border: 1px solid gray; background-color: #2a2a2a;");
    groupsImageLabel_->setMinimumSize(400, 400);
    groupsImageLabel_->setText("Run analysis to see edge groups");
    QScrollArea* groupsScroll = new QScrollArea();
    groupsScroll->setWidget(groupsImageLabel_);
    groupsScroll->setWidgetResizable(true);
    groupsLayout->addWidget(groupsScroll);
    imageTabWidget_->addTab(groupsTab, "Edge Groups");

    // Вкладка с картой размещения
    QWidget* placementTab = new QWidget();
    QVBoxLayout* placementLayout = new QVBoxLayout(placementTab);
    placementImageLabel_ = new QLabel();
    placementImageLabel_->setAlignment(Qt::AlignCenter);
    placementImageLabel_->setStyleSheet("border: 1px solid gray; background-color: #2a2a2a;");
    placementImageLabel_->setMinimumSize(400, 400);
    placementImageLabel_->setText("Run synthesis to see placement map");
    QScrollArea* placementScroll = new QScrollArea();
    placementScroll->setWidget(placementImageLabel_);
    placementScroll->setWidgetResizable(true);
    placementLayout->addWidget(placementScroll);
    imageTabWidget_->addTab(placementTab, "Placement Map");

    // Вкладка с результатом
    QWidget* resultTab = new QWidget();
    QVBoxLayout* resultLayout = new QVBoxLayout(resultTab);
    resultImageLabel_ = new QLabel();
    resultImageLabel_->setAlignment(Qt::AlignCenter);
    resultImageLabel_->setStyleSheet("border: 1px solid gray; background-color: #2a2a2a;");
    resultImageLabel_->setMinimumSize(400, 400);
    resultImageLabel_->setText("Run synthesis to see result");
    QScrollArea* resultScroll = new QScrollArea();
    resultScroll->setWidget(resultImageLabel_);
    resultScroll->setWidgetResizable(true);
    resultLayout->addWidget(resultScroll);
    imageTabWidget_->addTab(resultTab, "Output Texture");
}

void MainWindow::createLogPanel() {
    QDockWidget* logDock = new QDockWidget("Log", this);
    logDock->setAllowedAreas(Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);

    logTextEdit_ = new QTextEdit();
    logTextEdit_->setReadOnly(true);
    logTextEdit_->setFont(QFont("Consolas", 9));

    logDock->setWidget(logTextEdit_);
    addDockWidget(Qt::BottomDockWidgetArea, logDock);
}

void MainWindow::updateButtonStates() {
    bool hasImage = controller_->hasImage();
    bool canAnalyze = hasImage && !isAnalyzing_ && !isSynthesizing_;
    bool canSynthesize = hasImage && !isAnalyzing_ && !isSynthesizing_ && isAnalyzed_;
    bool canSave = !isAnalyzing_ && !isSynthesizing_;

    loadImageButton_->setEnabled(!isAnalyzing_ && !isSynthesizing_);
    analyzeButton_->setEnabled(canAnalyze);
    synthesizeButton_->setEnabled(canSynthesize && controller_->hasImage());
    cancelButton_->setEnabled(isAnalyzing_ || isSynthesizing_);
    saveResultButton_->setEnabled(canSave && !controller_->getOutputTexture().empty());
    savePlacementButton_->setEnabled(canSave && !controller_->getPlacementMap().empty());

    // Обновляем состояние панелей параметров
    analysisParamsGroup_->setEnabled(!isAnalyzing_ && !isSynthesizing_);
    synthesisParamsGroup_->setEnabled(!isAnalyzing_ && !isSynthesizing_);
}

QPixmap MainWindow::cvMatToQPixmap(const cv::Mat& mat) {
    if (mat.empty()) {
        return QPixmap();
    }

    cv::Mat display;
    cv::Mat rgb;

    // Конвертируем в RGB если нужно
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        display = rgb;
    }
    else if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
        display = rgb;
    }
    else {
        display = mat;
    }

    QImage img(display.data, display.cols, display.rows,
        display.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img);
}

bool MainWindow::saveImageViaQt(const cv::Mat& image, const QString& filePath) {
    if (image.empty()) {
        return false;
    }

    cv::Mat rgb;

    if (image.channels() == 3) {
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    }
    else if (image.channels() == 1) {
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    }
    else {
        return false;
    }

    QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    QImage copy = qimg.copy();

    return copy.save(filePath);
}

void MainWindow::displayImage(const cv::Mat& image, QLabel* label, int maxWidth) {
    if (image.empty()) {
        label->setText("No image");
        label->setPixmap(QPixmap());
        return;
    }

    QPixmap pixmap = cvMatToQPixmap(image);

    // Масштабируем если нужно
    if (pixmap.width() > maxWidth) {
        pixmap = pixmap.scaledToWidth(maxWidth, Qt::SmoothTransformation);
    }

    label->setPixmap(pixmap);
    //label->setFixedSize(pixmap.size());
    label->setText(""); // Убираем текст-заглушку
}

// ======================== Слоты действий пользователя ========================

void MainWindow::onLoadImage() {
    QString fileName = QFileDialog::getOpenFileName(this,
        "Select Input Image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)");

    if (fileName.isEmpty()) {
        return;
    }

    statusLabel_->setText("Loading image...");

    if (controller_->loadImage(fileName)) {
        cv::Mat original = controller_->getOriginalImage();
        displayImage(original, originalImageLabel_);
        statusLabel_->setText("Image loaded: " + fileName);

        // Очищаем другие вкладки
        groupsImageLabel_->setText("Run analysis to see edge groups");
        groupsImageLabel_->setPixmap(QPixmap());
        placementImageLabel_->setText("Run synthesis to see placement map");
        placementImageLabel_->setPixmap(QPixmap());
        resultImageLabel_->setText("Run synthesis to see result");
        resultImageLabel_->setPixmap(QPixmap());

        // Активируем кнопку анализа
        isAnalyzed_ = false;
        updateButtonStates();
    }
    else {
        statusLabel_->setText("Failed to load image");
        QMessageBox::warning(this, "Error", "Failed to load image");
    }

    updateButtonStates();
}

void MainWindow::onAnalyze() {
    if (!controller_->hasImage()) {
        QMessageBox::warning(this, "Warning", "Please load an image first");
        return;
    }

    isAnalyzing_ = true;
    updateButtonStates();

    // Очищаем предыдущие результаты
    groupsImageLabel_->setText("Analyzing...");
    groupsImageLabel_->setPixmap(QPixmap());

    statusLabel_->setText("Analyzing texture...");

    controller_->analyze();
}

void MainWindow::onSynthesize() {
    isSynthesizing_ = true;
    updateButtonStates();

    // Очищаем предыдущие результаты
    placementImageLabel_->setText("Synthesizing placement...");
    placementImageLabel_->setPixmap(QPixmap());
    resultImageLabel_->setText("Synthesizing texture...");
    resultImageLabel_->setPixmap(QPixmap());

    statusLabel_->setText("Synthesizing texture...");

    controller_->synthesize();
}

void MainWindow::onCancel() {
    controller_->cancel();
    statusLabel_->setText("Cancelling...");
}

void MainWindow::onSaveResult() {
    cv::Mat result = controller_->getOutputTexture();
    if (result.empty()) {
        QMessageBox::warning(this, "Warning", "No result to save");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this,
        "Save Output Texture",
        "output_texture.png",
        "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)");

    if (!fileName.isEmpty()) {
        if (saveImageViaQt(result, fileName)) {
            statusLabel_->setText("Result saved to " + fileName);
            onLogMessage("Result saved to " + fileName);
        }
        else {
            statusLabel_->setText("Failed to save result");
            QMessageBox::warning(this, "Error", "Failed to save image");
        }
    }
}

void MainWindow::onSavePlacement() {
    cv::Mat placement = controller_->getPlacementMap();
    if (placement.empty()) {
        QMessageBox::warning(this, "Warning", "No placement map to save");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this,
        "Save Placement Map",
        "placement_map.png",
        "PNG (*.png);;JPEG (*.jpg *.jpeg)");

    if (!fileName.isEmpty()) {
        if (saveImageViaQt(placement, fileName)) {
            statusLabel_->setText("Placement map saved to " + fileName);
            onLogMessage("Placement map saved to " + fileName);
        }
        else {
            statusLabel_->setText("Failed to save placement map");
            QMessageBox::warning(this, "Error", "Failed to save image");
        }
    }
}

// ======================== Обработка сигналов от контроллера ========================

void MainWindow::onAnalysisFinished(bool success, const QString& message) {
    isAnalyzing_ = false;

    if (success) {
        statusLabel_->setText("Analysis completed successfully");
        isAnalyzed_ = true;
    }
    else {
        statusLabel_->setText("Analysis failed: " + message);
        groupsImageLabel_->setText("Analysis failed\n\n" + message);
    }
    updateButtonStates();

    progressBar_->setVisible(false);
    progressBar_->setValue(0);
}

void MainWindow::onSynthesisFinished(bool success, const QString& message) {
    isSynthesizing_ = false;
    updateButtonStates();

    if (success) {
        statusLabel_->setText("Synthesis completed successfully");
        saveResultButton_->setEnabled(true);
        savePlacementButton_->setEnabled(true);
    }
    else {
        statusLabel_->setText("Synthesis failed: " + message);
    }

    progressBar_->setVisible(false);
    progressBar_->setValue(0);
}

void MainWindow::onLogMessage(const QString& message) {
    logTextEdit_->append(message);
    // Автопрокрутка вниз
    QTextCursor cursor = logTextEdit_->textCursor();
    cursor.movePosition(QTextCursor::End);
    logTextEdit_->setTextCursor(cursor);
}

void MainWindow::onAnalysisProgress(int percent) {
    progressBar_->setVisible(true);
    progressBar_->setValue(percent);
    statusLabel_->setText(QString("Analyzing... %1%").arg(percent));
}

void MainWindow::onSynthesisProgress(int percent) {
    progressBar_->setVisible(true);
    progressBar_->setValue(percent);
    statusLabel_->setText(QString("Synthesizing... %1%").arg(percent));
}

void MainWindow::onAnalysisGroupsVisualization(const cv::Mat& visualization) {
    displayImage(visualization, groupsImageLabel_);
    // Переключаемся на вкладку с группами, чтобы показать пользователю
    imageTabWidget_->setCurrentIndex(1);
}

void MainWindow::onSynthesisPlacementUpdated(const cv::Mat& placementMap) {
    displayImage(placementMap, placementImageLabel_);
}

void MainWindow::onSynthesisTextureUpdated(const cv::Mat& texture) {
    displayImage(texture, resultImageLabel_);
    // Переключаемся на вкладку с результатом, когда он готов
    imageTabWidget_->setCurrentIndex(3);
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (isAnalyzing_ || isSynthesizing_) {
        controller_->cancel();
        // Даем время на отмену
        QThread::msleep(500);
    }
    event->accept();
}
