#include <vector>
#include <iostream>
#include <Eigen/Dense>

int main(int argc, char** argv) {
    Eigen::Quaterniond quat;
    quat.w() = 0;
    quat.x() = 0;
    quat.y() = 1;
    quat.z() = 0;
    std::cerr << quat.toRotationMatrix() << "\n";
    return 0;
}
