#include <QApplication>
#include "MainWindow.h"


int main(int argc, char* argv[]){
    QApplication app(argc, argv);
    const std::string base_path = CONV_HOME;
    const std::string net_path = base_path + "data/net_.conv";
    MainWindow w;
    w.set_net_(net_path);
    w.show();
    return app.exec();
}
