#pragma once
#include <string>
#include <Eigen/Core>
#include <vector>

namespace ark {
    /** Gaussian Mixture Model */
    struct GaussianMixture {
        /** load Gaussian Mixture parameters from 'path' */
        void load(const std::string & path);

        /** get number of Gaussian mixture components */
        int numComponents() const;

        /** Compute PDF at 'input' */
        float pdf(const Eigen::Matrix<float, Eigen::Dynamic, 1> & x) const;

        /** Compute Ceres residual vector (squaredNorm of output vector is equal to min_i -log(c_i pdf_i(x))) */
        Eigen::Matrix<float, Eigen::Dynamic, 1> residual(const Eigen::Matrix<float, Eigen::Dynamic, 1> & x);

        /** Number of GMM components */
        int nComps;

        /** Number of dimensions data resides in */
        int nDims;

        /** Weight of each GMM component */
        Eigen::VectorXf weight;

        /** Mean of each GMM component */
        Eigen::MatrixXf mean;

        /** Covariance of each GMM component */
        std::vector<Eigen::MatrixXf> cov;

        // leading constants
        Eigen::VectorXd consts, consts_log;
        // cholesky decomposition of cov: cov = cov_cho * cov_cho^T
        std::vector<Eigen::MatrixXf> cov_cho;
        // cholesky decomposition of inverse: cov^-1 = covi_cho * covi_cho^T
        mutable std::vector<Eigen::MatrixXf> covi_cho;
    };
}  // namespace ark
