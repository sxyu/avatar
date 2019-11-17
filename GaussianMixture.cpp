#include "GaussianMixture.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

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
        float sqrt_2_pi_n = std::pow(2 * M_PI, nDims * 0.5 );
        float log_sqrt_2_pi_n = nDims * 0.5 * std::log(2 * M_PI);
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
        typedef Eigen::LLT<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> Cholesky;

        cov.resize(nComps);
        cov_cho.resize(nComps);
        prec_cho.resize(nComps);
        float minDet = std::numeric_limits<float>::max();
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
            float det = cov_cho[i].determinant();
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
    float GaussianMixture::pdf(const Eigen::VectorXf & x) const {
        float prob = 0.0f;
        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Mattype;
        for (int i = 0; i < nComps; ++i) {
            Eigen::TriangularView<Eigen::MatrixXf, Eigen::Lower> L(prec_cho[i]);
            auto residual = (L * (x - mean.row(i).transpose()));
            prob += consts[i] * std::exp(-0.5 * residual.squaredNorm());
        }
        return prob;
    }

    Eigen::VectorXf GaussianMixture::residual(const Eigen::VectorXf & x, int* comp_idx) const {
        float bestProb = std::numeric_limits<float>::max();
        Eigen::VectorXf ans(nDims + 1);
        Eigen::VectorXf residual(nDims + 1);
        residual[nDims] = 0.;
        for (int i = 0; i < nComps; ++i) {
            Eigen::TriangularView<Eigen::MatrixXf, Eigen::Lower> L(prec_cho[i]);
            residual.head(nDims) = L.transpose() * (x - mean.row(i).transpose()) * sqrt(0.5f);
            float p = residual.squaredNorm() - consts_log[i];
            if (p < bestProb) {
                bestProb = p;
                ans.noalias() = residual;
                *comp_idx = i;
            }
        }
        ans[nDims] = sqrt(-consts_log[*comp_idx]);
        return ans;
    }

    Eigen::VectorXf GaussianMixture::sample() const {
        // Pick random GMM component
        float randf = random_util::uniform(0.0f, 1.0f);
        int component;
        for (size_t i = 0 ; i < nComps; ++i) {
            randf -= weight[i];
            if (randf <= 0) component = i;
        }
        Eigen::VectorXf r(nDims);
        // Sample from Gaussian
        for (int i = 0; i < nDims; ++i) {
            r(i) = random_util::randn();
        }
        r *= cov_cho[component];
        r += mean.row(component);
        return r;
    }

}  // namespace ark
