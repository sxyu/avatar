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
        double pdf(const Eigen::VectorXd & x) const;

        /** Compute Ceres residual vector (squaredNorm of output vector is equal to min_i -log(c_i pdf_i(x))) */
        Eigen::VectorXd residual(const Eigen::VectorXd & x) const;

        /** Get a random sample from this distribution */
        Eigen::VectorXd sample() const;

        /** Number of GMM components */
        int nComps;

        /** Number of dimensions data resides in */
        int nDims;

        /** Weight of each GMM component */
        Eigen::VectorXd weight;

        /** Mean of each GMM component */
        Eigen::MatrixXd mean;

        /** Covariance of each GMM component */
        std::vector<Eigen::MatrixXd> cov;

        // leading constants
        Eigen::VectorXd consts, consts_log;
        // cholesky decomposition of cov: cov = cov_cho * cov_cho^T
        std::vector<Eigen::MatrixXd> cov_cho;
        // cholesky decomposition of inverse: cov^-1 = covi_cho * covi_cho^T
        mutable std::vector<Eigen::MatrixXd> covi_cho;
    };
}  // namespace ark
