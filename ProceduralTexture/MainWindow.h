#pragma once

#include <QMainWindow>
#include <memory>
#include <opencv2/opencv.hpp>

class AppController;
class QLabel;
class QTextEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QGroupBox;
class QTabWidget;
class QScrollArea;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    // Действия пользователя
    void onLoadImage();
    void onAnalyze();
    void onSynthesize();
    void onCancel();
    void onSaveResult();
    void onSavePlacement();

    // Обработка сигналов от контроллера
    void onAnalysisFinished(bool success, const QString& message);
    void onSynthesisFinished(bool success, const QString& message);
    void onLogMessage(const QString& message);
    void onAnalysisProgress(int percent);
    void onSynthesisProgress(int percent);
    void onAnalysisGroupsVisualization(const cv::Mat& visualization);
    void onSynthesisPlacementUpdated(const cv::Mat& placementMap);
    void onSynthesisTextureUpdated(const cv::Mat& texture);

    void onLargeFillChanged(int value);
    void onMediumFillChanged(int value);
    void onSmallFillChanged(int value);

private:
    void setupUi();
    void createMenuBar();
    void createToolBar();
    void createCentralWidget();
    void createParametersPanel();
    void createImageWidgets();
    void createLogPanel();
    void updateButtonStates();
    void updateParameterVisibility();

    void displayImage(const cv::Mat& image, QLabel* label, int maxWidth = 700);
    QPixmap cvMatToQPixmap(const cv::Mat& mat);

    std::unique_ptr<AppController> controller_;

    // Виджеты для отображения изображений
    QLabel* originalImageLabel_;
    QLabel* groupsImageLabel_;
    QLabel* placementImageLabel_;
    QLabel* resultImageLabel_;
    QTabWidget* imageTabWidget_;

    // Логирование и прогресс
    QTextEdit* logTextEdit_;
    QProgressBar* progressBar_;
    QLabel* statusLabel_;

    // Кнопки действий
    QPushButton* loadImageButton_;
    QPushButton* analyzeButton_;
    QPushButton* synthesizeButton_;
    QPushButton* cancelButton_;
    QPushButton* saveResultButton_;
    QPushButton* savePlacementButton_;

    // Панели параметров
    QGroupBox* analysisParamsGroup_;
    QGroupBox* synthesisParamsGroup_;

    // Параметры анализа
    QSpinBox* minEdgeLengthSpin_;
    QSpinBox* superpixelSizeSpin_;
    QDoubleSpinBox* thresholdSpin_;
    QDoubleSpinBox* rulerSpin_;

    // Параметры синтеза
    QSpinBox* outputWidthSpin_;
    QSpinBox* outputHeightSpin_;
    QCheckBox* rotationCheckBox_;
    QDoubleSpinBox* angleSpreadSpin_;
    QSpinBox* randomSeedSpin_;
    QDoubleSpinBox* densitySpin_;
    QDoubleSpinBox* scaleSpin_;

    QSpinBox* largeFillSpin_;
    QSpinBox* mediumFillSpin_;
    QSpinBox* smallFillSpin_;

    // Состояние
    bool isAnalyzed_ = false;
    bool isAnalyzing_ = false;
    bool isSynthesizing_ = false;
};
