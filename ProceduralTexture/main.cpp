#include "mainwindow.h"
#include <QApplication>

int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "ru");
    QApplication app(argc, argv);
    MainWindow window;
    window.showMaximized();
    return app.exec();
}