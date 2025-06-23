#ifndef CONV_LIB_GUI_EXAMPLE_MAINWINDOW_H
#define CONV_LIB_GUI_EXAMPLE_MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QVBoxLayout>
#include <simple_conv.h>

class QGraphicsView;
class QPixmap;
class QGraphicsPixmapItem;
class QWidget;
class QVBoxLayout;

class MainWindow : public QMainWindow {
Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void onImageChanged();
    void updateUI(const std::vector<float>& results);

private:
    simple_conv::net net_;
    QGraphicsPixmapItem* pixmapItem;
    QGraphicsScene* scene;
    QWidget* resultWidget;
    QVBoxLayout* resultLayout;
};

#endif //CONV_LIB_GUI_EXAMPLE_MAINWINDOW_H
