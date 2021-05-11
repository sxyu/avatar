#include "GaussianMixture.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <cmath>

#include "Util.h"

namespace ark {
    void GaussianMixture::load(const std::string & path)
    {
        std::ifstream ifs(path);
        if (!ifs) {
            std::cerr << "Warning: pose prior file at " << path << " does not exist or cannot be read\n";
            nComps = -1;
            return;
        }
        ifs >> nComps >> nDims;

        // compute constants
        double sqrt_2_pi_n = std::pow(2 * M_PI, nDims * 0.5 );
        double log_sqrt_2_pi_n = nDims * 0.5 * std::log(2 * M_PI);
        weight.resize(nComps);
        consts.resize(nComps);
        consts_log.resize(nComps);
        for (int i = 0; i < nComps; ++i) {
            // load weights
            ifs >> weight[i];
            consts_log[i] = log(weight[i]) - log_sqrt_2_pi_n;
            consts[i] = weight[i] / sqrt_2_pi_n;
        }

        mean.resize(nComps, nDims);
        for (int i = 0; i < nComps; ++i) {
            for (int j = 0; j < nDims; ++j) {
                // load mean vectors
                ifs >> mean(i, j);
            }
        }

        /** Cholesky decomposition */
        typedef Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> Cholesky;

        cov.resize(nComps);
        cov_cho.resize(nComps);
        prec_cho.resize(nComps);
        double minDet = std::numeric_limits<double>::max();
        for (int i = 0; i < nComps; ++i) {
            auto & m = cov[i];
            m.resize(nDims, nDims);
            for (int j = 0; j < nDims; ++j) {
                for (int k = 0; k < nDims; ++k) {
                    // load covariance matrices
                    ifs >> m(j, k);
                }
            }
            Cholesky chol(cov[i]);
            if (chol.info() != Eigen::Success) throw "Decomposition failed!";
            cov_cho[i] = chol.matrixL();
            Cholesky chol_prec(cov[i].inverse());
            prec_cho[i] = chol_prec.matrixL();
            double det = cov_cho[i].determinant();
            minDet = std::min(det, minDet);

            // update constants
            consts[i] /= det;
            consts_log[i] -= log(det);
        }

        for (int i = 0; i < nComps; ++i) {
            // normalize constants
            consts[i] *= minDet;
            consts_log[i] += log(minDet);
        }
    }

    int GaussianMixture::numComponents() const {
        return nComps;
    };

    /** Compute PDF at 'input' */
    double GaussianMixture::pdf(const Eigen::VectorXd & x) const {
        double prob(0.0);
        typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Mattype;
        for (int i = 0; i < nComps; ++i) {
            Eigen::TriangularView<Eigen::MatrixXd, Eigen::Lower> L(prec_cho[i]);
            auto residual = (L * (x - mean.row(i).transpose()));
            prob += consts[i] * std::exp(-0.5 * residual.squaredNorm());
        }
        return prob;
    }

    Eigen::VectorXd GaussianMixture::residual(const Eigen::VectorXd & x, int* comp_idx) const {
        double bestProb = std::numeric_limits<double>::max();
        Eigen::VectorXd ans;
        for (int i = 0; i < nComps; ++i) {
            Eigen::TriangularView<Eigen::MatrixXd, Eigen::Lower> L(prec_cho[i]);
            Eigen::VectorXd residual(nDims + 1);
            residual[nDims] = 0.;
            residual.head(nDims) = L.transpose() * (x - mean.row(i).transpose()) * sqrt(0.5);
            double p = residual.squaredNorm() - consts_log[i];
            if (p < bestProb) {
                bestProb = p;
                residual[nDims] = sqrt(-consts_log[i]);
                ans = residual;
                if (comp_idx != nullptr) {
                    *comp_idx = i;
                }
            }
        }
        return ans;
    }

    Eigen::VectorXd GaussianMixture::sample() const {
        // Pick random GMM component
        double randf = random_util::uniform(0.0f, 1.0f);
        int component;
        for (size_t i = 0 ; i < nComps; ++i) {
            randf -= weight[i];
            if (randf <= 0) component = i;
        }
        Eigen::VectorXd r(nDims);
        // Sample from Gaussian
        for (int i = 0; i < nDims; ++i) {
            r(i) = random_util::randn();
        }
        r *= cov_cho[component];
        r += mean.row(component);
        return r;
    }

}  // namespace ark
