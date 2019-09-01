#include "GaussianMixture.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <fstream>

namespace ark {
    void GaussianMixture::load(const std::string & path)
    {
        std::ifstream ifs(path);
        ifs >> nComps >> nDims;

        // compute constants
        float sqrt_2_pi_n = ceres::pow(2 * M_PI, nDims * 0.5 );
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
        covi_cho.resize(nComps);
        float maxDet = 0.0;
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
            covi_cho[i] = chol.matrixL().solve(Eigen::MatrixXf::Identity(nDims, nDims)).transpose();
            float det = covi_cho[i].determinant();
            maxDet = std::max(det, maxDet);

            // update constants
            consts[i] *= det;
            consts_log[i] += log(det);
        }

        for (int i = 0; i < nComps; ++i) {
            // normalize constants
            consts[i] /= maxDet;
            consts_log[i] -= log(maxDet);
        }
    }

    int GaussianMixture::numComponents() const {
        return nComps;
    };

    /** Compute PDF at 'input' */
    float GaussianMixture::pdf(const Eigen::Matrix<float, Eigen::Dynamic, 1> & x) const {
        float prob(0.0);
        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Mattype;
        for (int i = 0; i < nComps; ++i) {
            Eigen::TriangularView<Eigen::MatrixXf, Eigen::Lower> L(covi_cho[i]);
            auto residual = (L.transpose() * (x - mean.row(i).transpose()));
            prob += consts[i] * ceres::exp(-0.5 * residual.squaredNorm());
        }
        return prob;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 1> GaussianMixture::residual(const Eigen::Matrix<float, Eigen::Dynamic, 1> & x) {
        float bestProb = float(0);
        typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VecType;
        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatType;
        VecType ans;
        for (int i = 0; i < nComps; ++i) {
            Eigen::TriangularView<Eigen::MatrixXf, Eigen::Lower> L(covi_cho[i]);
            VecType residual(nDims + 1);
            residual[nDims] = float(0);
            residual.head(nDims) = L.transpose() * (x - mean.row(i).transpose()) * sqrt(0.5);
            float p = residual.squaredNorm() - float(consts_log[i]);
            if (p < bestProb || !i) {
                bestProb = p;
                residual[nDims] = float(sqrt(-consts_log[i]));
                ans = residual;
            }
        }
        return ans;
    }

}  // namespace ark
