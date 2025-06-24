#include "MainWindow.h"
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QPixmap>
#include <QMouseEvent>
#include <QPainter>
#include <QLabel>
#include <QProgressBar>
#include <QScrollArea>
#include <QVBoxLayout>
#include <vector>
#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <opencv4/opencv2/opencv.hpp>

std::vector<float> forward(const QPixmap& pixmap, simple_conv::net& net_) {

    QImage image = pixmap.toImage();
    if(image.format() != QImage::Format_RGB888)
        image = image.convertToFormat(QImage::Format_RGB888);

    cv::Mat cv_img(image.height(), image.width(), CV_8UC3, const_cast<uchar*>(image.bits()), image.bytesPerLine());
    try {
        simple_conv::preprocessing::crop_image(cv_img, true);
    }catch (const std::exception& e){
        return {};
    }

    cv_img = cv_img.reshape(1, (int) cv_img.total());

    auto result = simple_conv::forward(cv_img, net_);

    std::vector<float> ret(result.size());
    memcpy(ret.data(), result.data, result.size() * sizeof(float));

    return ret;
}

class PaintablePixmapItem : public QGraphicsPixmapItem {
public:
    PaintablePixmapItem(const QPixmap& pixmap, QObject* parent = nullptr)
            : QGraphicsPixmapItem(pixmap), parentWindow(parent) {}

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override {
        if((event->modifiers() & Qt::ControlModifier) && event->button() == Qt::RightButton){
            clear();
        }else {
            drawAt(event->pos(), event->button());
        }
    }

    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override {
        if(event->buttons() & Qt::LeftButton){
            drawAt(event->pos(), Qt::LeftButton);
        }else if(event->buttons() & Qt::RightButton){
            drawAt(event->pos(), Qt::RightButton);
        }
    }

private:
    QObject* parentWindow;

    void clear() {
        QPixmap newPixmap = pixmap().copy();
        newPixmap.fill(Qt::black);
        setPixmap(newPixmap);
        update();
    }

    void drawAt(const QPointF& pos, Qt::MouseButton button) {
        QPixmap pix = pixmap();
        QPainter painter(&pix);
        QPen pen;
        if(button == Qt::LeftButton) {
            pen.setColor(Qt::white);
            pen.setWidth(25);
        }else if(button == Qt::RightButton) {
            pen.setWidth(25);
            pen.setColor(Qt::black);
        }else{
            pen.setColor(Qt::white);
        }
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        painter.setPen(pen);
        painter.drawPoint(pos);
        painter.end();

        setPixmap(pix);
        update();

        if (parentWindow) {
            QMetaObject::invokeMethod(parentWindow, "onImageChanged", Qt::QueuedConnection);
        }
    }
};

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {

    QPixmap pixmap(400, 400);
    pixmap.fill(Qt::black);

    scene = new QGraphicsScene(this);
    pixmapItem = new PaintablePixmapItem(pixmap, this);
    scene->addItem(pixmapItem);

    QGraphicsView* view = new QGraphicsView(scene);

    QScrollArea* scrollArea = new QScrollArea();
    resultWidget = new QWidget();
    resultLayout = new QVBoxLayout(resultWidget);
    resultLayout->setAlignment(Qt::AlignTop);
    resultLayout->addStretch();

    resultWidget->setLayout(resultLayout);
    scrollArea->setWidget(resultWidget);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFixedWidth(300);
    scrollArea->setFixedHeight(550);

    QWidget* centralWidget = new QWidget(this);
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);

    mainLayout->addWidget(view);
    mainLayout->addWidget(scrollArea);

    centralWidget->setLayout(mainLayout);
    setCentralWidget(centralWidget);

    connect(this, &MainWindow::onImageChanged, this, &MainWindow::onImageChanged);
}

void MainWindow::onImageChanged() {
    std::vector<float> results = forward(pixmapItem->pixmap(), net_);
    updateUI(results);
}

void MainWindow::updateUI(const std::vector<float>& results) {
    QLayoutItem* item;
    while ((item = resultLayout->takeAt(0)) != nullptr) {
        delete item->widget();
        delete item;
    }

    for (size_t i = 0; i < results.size(); ++i) {
        QLabel* label = new QLabel(QString("Result %1").arg(i));
        QProgressBar* bar = new QProgressBar();
        bar->setRange(0, 100);
        bar->setValue(static_cast<int>(results[i] * 100));

        resultLayout->addWidget(label);
        resultLayout->addWidget(bar);
    }
}
