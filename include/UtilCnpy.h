#pragma once

#include <Eigen/Core>
#include <cnpy.h>

namespace ark {
namespace util {

// Matrix load helper; currently copies on return
// can modify cnpy to load into the Eigen matrix; not important for now
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
loadFloatMatrix(const cnpy::NpyArray& raw, size_t r, size_t c);

Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
loadUintMatrix(const cnpy::NpyArray& raw, size_t r, size_t c);

const size_t ANY_SHAPE = (size_t)-1;
void assertShape(const cnpy::NpyArray& m, std::initializer_list<size_t> shape);

}  // namespace util
}  // namespace ark
