#include "AvatarOptimizer.h"

#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <Eigen/StdVector>
#include <ceres/ceres.h>
#include <nanoflann.hpp>
#include <boost/thread.hpp>

#include "Version.h"
#include "Avatar.h"
#include "AvatarRenderer.h"
#include "Util.h"

// #define PCL_DEBUG_VISUALIZE

#ifndef OPENARK_PCL_ENABLED
// If PCL not available then cannot visualize
#undef PCL_DEBUG_VISUALIZE
#endif

#ifdef PCL_DEBUG_VISUALIZE
#include <pcl/visualization/pcl_visualizer.h>
#endif

#define BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define PROFILE(x) do{printf("%s: %f ms\n", #x, std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count()); start = std::chrono::high_resolution_clock::now(); }while(false)

// Uncomment below line to compare analytic diff results to auto diff
//#define TEST_COMPARE_AUTO_DIFF

namespace nanoflann {
    /// KD-tree adaptor for working with data directly stored in a column-major Eigen Matrix, without duplicating the data storage.
    /// This code is adapted from the KDTreeEigenMatrixAdaptor class of nanoflann.hpp
    template <class MatrixType, int DIM = -1, class Distance = nanoflann::metric_L2_Simple, typename IndexType = int>
        struct KDTreeEigenColMajorMatrixAdaptor {
            typedef KDTreeEigenColMajorMatrixAdaptor<MatrixType, DIM, Distance> self_t;
            typedef typename MatrixType::Scalar              num_t;
            typedef typename Distance::template traits<num_t,self_t>::distance_t metric_t;
            typedef KDTreeSingleIndexAdaptor< metric_t,self_t,DIM,IndexType>  index_t;
            index_t* index;
            KDTreeEigenColMajorMatrixAdaptor(const MatrixType &mat, const int leaf_max_size = 10) : m_data_matrix(mat) {
                const size_t dims = mat.rows();
                index = new index_t(dims, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
                index->buildIndex();
            }
            ~KDTreeEigenColMajorMatrixAdaptor() {delete index;}
            const MatrixType &m_data_matrix;
            /// Query for the num_closest closest points to a given point (entered as query_point[0:dim-1]).
            inline void query(const num_t *query_point, const size_t num_closest, IndexType *out_indices, num_t *out_distances_sq) const {
                nanoflann::KNNResultSet<typename MatrixType::Scalar,IndexType> resultSet(num_closest);
                resultSet.init(out_indices, out_distances_sq);
                index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
            }
            /// Query for the closest points to a given point (entered as query_point[0:dim-1]).
            inline IndexType closest(const num_t *query_point) const {
                IndexType out_indices;
                num_t out_distances_sq;
                query(query_point, 1, &out_indices, &out_distances_sq);
                return out_indices;
            }
            const self_t & derived() const {return *this;}
            self_t & derived() {return *this;}
            inline size_t kdtree_get_point_count() const {return m_data_matrix.cols();}
            /// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
            inline num_t kdtree_distance(const num_t *p1, const size_t idx_p2,size_t size) const {
                num_t s=0;
                for (size_t i=0; i<size; i++) {
                    const num_t d= p1[i]-m_data_matrix.coeff(i,idx_p2);
                    s+=d*d; }
                return s;
            }
            /// Returns the dim'th component of the idx'th point in the class:
            inline num_t kdtree_get_pt(const size_t idx, int dim) const {
                return m_data_matrix.coeff(dim,idx);
            }
            /// Optional bounding-box computation: return false to default to a standard bbox computation loop.
            template <class BBOX> bool kdtree_get_bbox(BBOX&) const {return false;}
        };
}  // namespace nanoflann

namespace ceres {
    /** WHY THIS? For our use case it is desirable to pre-compute the full Jacobian at an earlier stage but
     *  make use of the special addition operator for local parameterization. This is because some
     *  Jacobians are easier to compute wrt the local 'error' vector than the state. This is a known Ceres
     *  limitation, see https://groups.google.com/forum/#!msg/ceres-solver/fs8iNI9_F7Q/gv0R4NGLAwAJ
     *  Here, we set Jacobian to a dummy identity matrix, as suggested by a post on that page. The
     *  local Jacobian is multiplied manually in AvatarEvaluationCommonData where required. */
    class FakeQuaternionParameterization : public ceres::LocalParameterization {
        public:
            ~FakeQuaternionParameterization() {}
            bool Plus(const double* x_ptr,
                    const double* delta,
                    double* x_plus_delta_ptr) const {
                // Copied from EigenQuaternionParameterization
                Eigen::Map<Eigen::Quaterniond> x_plus_delta(x_plus_delta_ptr);
                Eigen::Map<const Eigen::Quaterniond> x(x_ptr);

                const double norm_delta =
                    sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
                if (norm_delta > 0.0) {
                    const double sin_delta_by_delta = sin(norm_delta) / norm_delta;

                    // Note, in the constructor w is first.
                    Eigen::Quaterniond delta_q(cos(norm_delta),
                            sin_delta_by_delta * delta[0],
                            sin_delta_by_delta * delta[1],
                            sin_delta_by_delta * delta[2]);
                    x_plus_delta = delta_q * x;
                } else {
                    x_plus_delta = x;
                }
                return true;
            }
            bool ComputeJacobian(const double* x, double* jacobian) const {
                Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor> > J(jacobian);
                J.topLeftCorner<3,3>().setIdentity();
                J.bottomRows<1>().setZero();
                return true;
            }
            int GlobalSize() const { return 4; }
            int LocalSize() const { return 3; }
    };
}

namespace ark {
    namespace {
        using MatAlloc = Eigen::aligned_allocator<Eigen::Matrix3d>;
        using VecAlloc = Eigen::aligned_allocator<Eigen::Vector3d>;

        /** Evaluation callback for Ceres */
        template<class Cache>
            struct AvatarEvaluationCommonData : public ceres::EvaluationCallback {
                /** Max number of joint assignments to consider for each joint */
                static const int MAX_ASSIGN = 4;

                /** If true, will recalculate shape every update */
                const bool shapeEnabled;

                AvatarEvaluationCommonData(AvatarOptimizer& opt, bool shape_enabled = false)
                    : opt(opt), ava(opt.ava), numJointSpaces(opt.ava.model.numJoints() + 1), shapeEnabled(shape_enabled) {
                        // _L.resize(nJoints * AvatarOptimizer::ROT_SIZE);
                        localJacobian.resize(numJointSpaces);
                        if (shape_enabled) {
                            S.resize(ava.model.numJoints());
                            Sp.resize(ava.model.numJoints());
                            H.resize(ava.model.numJoints());
                        }
                        _R.resize(numJointSpaces * numJointSpaces);
                        _t.resize(_R.size());
                        shapedCloud.resize(3, ava.model.numPoints());

                        CalcShape();

                        // Make a list of deduplicated ancestor joints for each point
                        ancestor.resize(ava.model.numPoints());
                        for (int point = 0; point < ava.model.numPoints(); ++point) {
                            auto& ances = ancestor[point];
                            for (auto & weight_joint : ava.model.assignedJoints[point]) {
                                double weight = weight_joint.first;
                                int joint = weight_joint.second;
                                ances.emplace_back(joint, joint, weight);
                                for (int j = ava.model.parent[joint]; j != -1; j = ava.model.parent[j]) {
                                    ances.emplace_back(j, joint, weight);
                                }
                            }
                            std::sort(ances.begin(), ances.end());
                            int last = 0;
                            for (size_t i = 1; i < ances.size(); ++i) {
                                if (ances[last].jid == ances[i].jid) {
                                    ances[last].merge(ances[i]);
                                } else {
                                    ++last;
                                    if (last < i) {
                                        ances[last] = std::move(ances[i]);
                                    }
                                }
                            }
                            ances.resize(last + 1);
                        }

                        if (shape_enabled) {
                            for (int j = 0; j < ava.model.numJoints(); ++j) {
                                Sp[j].resize(3, ava.model.numShapeKeys());
                                S[j].resize(3, ava.model.numShapeKeys());
                                H[j].resize(3, ava.model.numShapeKeys());
                            }

                            if (ava.model.useJointShapeRegressor) {
                                for (int j = 0; j < ava.model.numJoints(); ++j) {
                                    S[j].noalias() = ava.model.jointShapeReg.middleRows<3>(3 * j);
                                }
                            } else {
                                for (int i = 0; i < ava.model.numShapeKeys(); ++i) {
                                    Eigen::MatrixXd::Scalar d;
                                    Eigen::Map<const CloudType> keyCloud(ava.model.keyClouds.data() + i * ava.model.numPoints() * 3,
                                            3, ava.model.numPoints());
                                    for (int j = 0; j < ava.model.numJoints(); ++j) {
                                        S[j].col(i).noalias() = keyCloud * ava.model.jointRegressor.col(j);
                                    }
                                }
                            }
                            for (int j = 1; j < ava.model.numJoints(); ++j) {
                                if (j) Sp[j].noalias() = S[j] - S[ava.model.parent[j]];
                            }
                            Sp[0].setZero();
                            H[0].setZero();
                        }
                    }

                /** Compute joint and point positions after applying shape keys*/
                void CalcShape() {
                    Eigen::Map<Eigen::VectorXd> shapedCloudVec(shapedCloud.data(), 3 * shapedCloud.cols());

                    /** Apply shape keys */
                    shapedCloudVec.noalias() = ava.model.keyClouds * ava.w + ava.model.baseCloud; 

                    /** Apply joint [shape] regressor */
                    // TODO: use dense joint regressor with compressed cloud
                    if (ava.model.useJointShapeRegressor) {
                        jointPosInit.resize(3, ava.model.numJoints());
                        Eigen::Map<Eigen::VectorXd> jointPosVec(jointPosInit.data(), 3 * ava.model.numJoints());
                        jointPosVec.noalias() = ava.model.jointShapeRegBase + ava.model.jointShapeReg * ava.w;
                    } else {
                        jointPosInit.noalias() = shapedCloud * ava.model.jointRegressor;
                    }

                    /** Ensure root is at origin*/
                    Eigen::Vector3d offset = jointPosInit.col(0);
                    shapedCloud.colwise() -= offset;
                    jointPosInit.colwise() -= offset;

                    /** Find relative positions of each joint */
                    jointVecInit.noalias() = jointPosInit;
                    for (int i = ava.model.numJoints() - 1; i >= 1; --i) {
                        jointVecInit.col(i).noalias() -= jointVecInit.col(ava.model.parent[i]);
                    }
                    shapeComputed = true;
                }

                void PrepareForEvaluation(bool evaluate_jacobians,
                        bool new_evaluation_point) final {
                    // std::cerr << "PREP " << evaluate_jacobians << ", " << new_evaluation_point << "\n";
                    if (new_evaluation_point) {
                        if (shapeEnabled || !shapeComputed) CalcShape();
                        jointVecInit.col(0).noalias() = ava.p;

                        for (int i = 0; i < numJointSpaces - 1; ++i) {
                            // double * jacobian = localJacobian[i].data();
                            auto& x = opt.r[i].coeffs();
                            /** Jacobian of local parameterization '+' function at 0 */
                            localJacobian[i] << x(3),  x(2), -x(1),
                                -x(2),  x(3),  x(0),
                                x(1), -x(0),  x(3),
                                -x(0), -x(1), -x(2);
                        }

                        // Compute relative rotations and translations
                        R(-1, -1).setIdentity(); // -1 means 'to global'
                        t(-1, -1).setZero();
                        for (int i = 0; i < numJointSpaces - 1; ++i) {
                            R(i, i).setIdentity();
                            Eigen::Matrix3d rot = opt.r[i].toRotationMatrix();
                            t(i, i).setZero();
                            int p = ava.model.parent[i];
                            for (int j = p; ; j = ava.model.parent[j]) {
                                R(j, i).noalias() = R(j, p) * rot;
                                t(j, i).noalias() = R(j, p) * jointVecInit.col(i) + t(j, p);
                                if (j == -1) break;
                            }
                        }

                        if (shapeEnabled) {
                            // Compute joint-to-parent accumulated shape differences
                            for (int j = 1; j < numJointSpaces - 1; ++j) {
                                H[j].noalias() = R(-1, ava.model.parent[j]) * Sp[j] + H[ava.model.parent[j]];
                            }
                        }

                        std::atomic<size_t> cacheId(0);
                        auto worker = [evaluate_jacobians, &cacheId, this](int workerId) {
                            size_t workerCacheId;
                            while (true) {
                                workerCacheId = cacheId++;
                                if (workerCacheId >= caches.size()) break;
                                caches[workerCacheId].updateData(evaluate_jacobians);
                            }
                        };

                        std::vector<boost::thread> threadPool;
                        for (int i = 0; i < numThreads; ++i) {
                            threadPool.emplace_back(worker, i);
                        }
                        for (auto& thread : threadPool) {
                            thread.join();
                        }
                    }
                }

                /** Joint-to-ancestor joint rotation (j_ances=-1 is global) */
                inline Eigen::Matrix3d& R(int j_ancestor, int j) {
                    return _R[numJointSpaces * (j_ancestor+1) + j+1];
                }
                /** Joint-to-ancestor joint relative position (j_ances=-1 is global) */
                inline Eigen::Vector3d& t(int j_ancestor, int j) {
                    return _t[numJointSpaces * (j_ancestor+1) + j+1];
                }

                /** Combined left-side matrix to multiply into Jacobians for joint j, component t */
                // inline Eigen::Matrix3d& L(int j, int t) {
                //     return _L[j * AvatarOptimizer::ROT_SIZE + t];
                // }
                AvatarOptimizer& opt;
                Avatar& ava;
                /** WARNING: is actual number of joints + 1 */
                int numJointSpaces;

                struct Ancestor {
                    Ancestor() {}
                    explicit Ancestor(int jid, int assigned = -1, double assign_weight = 0.0) : jid(jid) {
                        if (assigned >= 0) {
                            assign[0] = assigned;
                            weight[0] = assign_weight;
                            num_assign = 1;
                        } else {
                            num_assign = 0;
                        }
                    }

                    /** Joint ID */
                    int jid; 

                    /** Assigned joints of the skin point corresponding to the joint */ 
                    int assign[MAX_ASSIGN];

                    /** Assignment weight */
                    double weight[MAX_ASSIGN];

                    /** Number of assigned joints. */
                    int num_assign;

                    /** Compare by joint id */
                    bool operator <(const Ancestor& other) const {
                        return jid < other.jid;
                    }

                    /** Merge another ancestor joint */
                    void merge(const Ancestor& other) {
                        for (int i = 0; i < other.num_assign; ++i) {
                            weight[num_assign] = other.weight[i];
                            assign[num_assign++] = other.assign[i];
                        }
                    }
                };

                /** Max number of threads to use during common pre-evaluation computations */
                int numThreads = 1;

                /** INTERNAL: Scaled versions of betaPose, betaShape actually
                 * used in the optimization. This accounts for
                 * variations in the number of ICP-type residuals.  */
                double scaledBetaPose, scaledBetaShape;

                /** Deduped, topo sorted ancestor joints for each skin point,
                 *  combining all joint assignments for the point */
                std::vector<std::vector<Ancestor> > ancestor;

                /** Joint initial relative/absolute positions */
                CloudType jointVecInit, jointPosInit;

                /** baseCloud after applying shape keys (3 * num points) */
                CloudType shapedCloud;

                /** List of point-specific caches */
                std::vector<Cache> caches;

                /** Local parameterization jacobian */
                std::vector<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 3, Eigen::RowMajor> > > localJacobian;

                /** Joint-to-parent 'shape deltas' */
                std::vector<CloudType, Eigen::aligned_allocator<CloudType> > S;

                /** Joint-to-parent 'shape delta difference' */
                std::vector<CloudType, Eigen::aligned_allocator<CloudType> > Sp;

                /** Accumulated joint-to-parent 'shape delta difference' */
                std::vector<CloudType, Eigen::aligned_allocator<CloudType> > H;

                private:
                /** Joint-to-ancestor joint rotation (j_ances=-1 is global) */
                std::vector<Eigen::Matrix3d, MatAlloc> _R;

                /** Joint-to-ancestor joint relative position (j_ances=-1 is global) */
                std::vector<Eigen::Vector3d, VecAlloc> _t;

                /** Combined left-side matrix to use for Jacobians for joint j, component t */
                // std::vector<Eigen::Matrix3d, MatAlloc> _L;

                /** True if CalcShape() has been called, to avoid further calls */
                bool shapeComputed;
            };

        /** Common method for each model point */
        struct AvatarCostFunctorCache {
            AvatarCostFunctorCache(AvatarEvaluationCommonData<AvatarCostFunctorCache>& common_data,
                    int point_id)
                : commonData(common_data), opt(common_data.opt), ava(common_data.opt.ava),
                pointId(point_id) {
                    icpJacobian.resize(commonData.ancestor[pointId].size());
                    if(commonData.shapeEnabled) {
                        icpShapeJacobian.resize(3, ava.model.numShapeKeys());
                    }
                }

            bool getICPJacobians(double* residuals, double** jacobians) const {
                Eigen::Map<Eigen::Vector3d> residualMap(residuals);
                residualMap.noalias() = resid;
                if (jacobians != nullptr) {
                    if (jacobians[0] != nullptr) {
                        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                        J.setIdentity();
                    }
                    size_t i = 0;
                    for (; i < commonData.ancestor[pointId].size(); ++i) {
                        if (jacobians[i+1] != nullptr) {
                            Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > J(jacobians[i+1]);
#ifdef TEST_COMPARE_AUTO_DIFF
                            J.noalias() = icpJacobian[i];
#else
                            J.topLeftCorner<3,3>().noalias() = icpJacobian[i];
                            J.rightCols<1>().setZero();
#endif
                        }
                    }
                    if (commonData.shapeEnabled && jacobians[i + 1] != nullptr) {
                        Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> > J(jacobians[i + 1], 3, ava.model.numShapeKeys());
                        J.noalias() = icpShapeJacobian;
                    }
                }
                return true;
            }

            void updateData(bool compute_jacobians) {
                auto pointPosInit = commonData.shapedCloud.col(pointId);
                resid.setZero();
                for (auto& assign : ava.model.assignedJoints[pointId]) {
                    int k = assign.second;
                    resid += assign.first * (commonData.R(-1, k) * (pointPosInit - commonData.jointPosInit.col(k)) + commonData.t(-1, k));
                }
                if (compute_jacobians) {
                    // Root position derivative is always identity
                    // TODO: precompute point-to-assigned-joint vector, reduce 3 flops
                    //CloudType pointVecs;

                    Eigen::Matrix<double, 3, 4> dRot;
                    // Eigen::Matrix3d vCross;
                    Eigen::Vector3d v;

                    for (size_t i = 0; i < commonData.ancestor[pointId].size(); ++i) {
                        // Set derivative for each parent rotation
                        auto& ances = commonData.ancestor[pointId][i];
                        int j = ances.jid; // 'middle' joint we are differenting wrt

                        v.setZero();
                        for (int assign = 0; assign < ances.num_assign; ++assign) {
                            // up to 4 inner loops
                            int k = ances.assign[assign]; // 'outer' joint assigned to the point
                            v += ances.weight[assign]* (commonData.R(j, k) *
                                    (pointPosInit - commonData.jointPosInit.col(k)) + commonData.t(j, k));
                        }
                        // std::cerr <<v.transpose<< "\n"

                        Eigen::Quaterniond& q = opt.r[j];
                        Eigen::Vector3d u = q.vec() * 2;
                        // Quaternion-vector rotation (pseudo-)Jacobian
                        double w = q.w() * 2;
                        dRot << 
                            u(1)*v(1) + v(2)*u(2)    ,
                            w*v(2)    + u(0)*v(1)   - 2*u(1)*v(0),
                            -w*v(1)    - 2*v(0)*u(2) + u(0)*v(2),
                            u(1)*v(2) - v(1)*u(2),

                            -w*v(2)    - 2*u(0)*v(1) + v(0)*u(1),
                            v(2)*u(2) + u(0)*v(0),
                            w*v(0)    + u(1)*v(2)   - 2*v(1)*u(2),
                            v(0)*u(2) - u(0)*v(2),

                            w*v(1)    + v(0)*u(2)   - 2*u(0)*v(2),
                            -w*v(0)    - 2*u(1)*v(2) + v(1)*u(2),
                            u(0)*v(0) + v(1)*u(1),
                            u(0)*v(1) - v(0)*u(1);

                        icpJacobian[i].noalias() = commonData.R(-1, ava.model.parent[j]) * dRot
#ifndef TEST_COMPARE_AUTO_DIFF
                            * commonData.localJacobian[j]
#endif
                            ;
                    }

                    if (commonData.shapeEnabled) {
                        icpShapeJacobian.setZero();
                        for (const std::pair<double,int>& assign : ava.model.assignedJoints[pointId]) {
                            const int j = assign.second;
                            auto pointDeltas = ava.model.keyClouds.middleRows<3>(pointId * 3);
                            icpShapeJacobian += (commonData.R(-1, j) * (pointDeltas - commonData.S[j]) + commonData.H[j]) * assign.first;
                        }
                    }
                }
            }

            Eigen::Vector3d resid;
#ifdef TEST_COMPARE_AUTO_DIFF
            // For comparing with auto diff, we cannot multiply by the
            // local param jacobian or result would not be comparable
            std::vector<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>,
                Eigen::aligned_allocator<
                    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > > icpJacobian;
#else
            std::vector<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>,
                Eigen::aligned_allocator<
                    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > > icpJacobian;
#endif

            Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> icpShapeJacobian;

            Avatar& ava;
            AvatarOptimizer& opt;
            AvatarEvaluationCommonData<AvatarCostFunctorCache>& commonData;
            int pointId;
        };

        /** Ceres analytic derivative cost function for ICP error.
         *  Works for pose parameters only. */
        struct AvatarICPCostFunctor : ceres::CostFunction {
            AvatarICPCostFunctor(AvatarEvaluationCommonData<AvatarCostFunctorCache> & common_data,
                    size_t cache_id, const CloudType& data_cloud, int data_point_id)
                : commonData(common_data), cacheId(cache_id), dataCloud(data_cloud), dataPointId(data_point_id) {
                    set_num_residuals(3);
                    auto& cache = common_data.caches[cache_id];

                    std::vector<int> * paramBlockSizes = mutable_parameter_block_sizes();
                    paramBlockSizes->push_back(3); // Root position
                    for (size_t i = 0; i < cache.commonData.ancestor[cache.pointId].size(); ++i) {
                        paramBlockSizes->push_back(4); // Add rotation block for each ancestor
                    }
                    if (commonData.shapeEnabled) paramBlockSizes->push_back(cache.ava.model.numShapeKeys()); // Shape key weights?
                }

            bool Evaluate(double const* const* parameters,
                    double* residuals,
                    double** jacobians) const final {
                if (!commonData.caches[cacheId].getICPJacobians(residuals, jacobians)) return false;
                Eigen::Map<Eigen::Vector3d> resid(residuals);
                resid -= dataCloud.col(dataPointId);
                return true;
            }
            const CloudType& dataCloud;
            int dataPointId;
            size_t cacheId;
            AvatarEvaluationCommonData<AvatarCostFunctorCache>& commonData;
        };

        /** Ceres analytic derivative cost function for pose prior error */
        struct AvatarPosePriorCostFunctor : ceres::CostFunction {
            AvatarPosePriorCostFunctor(AvatarEvaluationCommonData<AvatarCostFunctorCache> & common_data)
                : commonData(common_data), posePrior(commonData.ava.model.posePrior),
                nSmplJoints(commonData.ava.model.numJoints() - 1) {
                    set_num_residuals(nSmplJoints * 3 + 1); // 3 for each joint + 1 extra
                    std::vector<int> * paramBlockSizes = mutable_parameter_block_sizes();
                    for (int i = 0; i < nSmplJoints; ++i) {
                        paramBlockSizes->push_back(4); // Add rotation block for each non-root joint
                    }
                }

            bool Evaluate(double const* const* parameters,
                    double* residuals,
                    double** jacobians) const final {
                const int nResids = nSmplJoints * 3 + 1;
                Eigen::VectorXd smplParams(nSmplJoints * 3);
                for (int i = 0; i < nSmplJoints; ++i) {
                    Eigen::Map<const Eigen::Quaterniond> q(parameters[i]);
                    Eigen::AngleAxisd aa(q);
                    smplParams.segment<3>(i * 3) = aa.axis() * aa.angle();
                }
                Eigen::Map<Eigen::VectorXd> resid(residuals, nResids);
                int compIdx;
                resid.noalias() = posePrior.residual(smplParams, &compIdx) * commonData.scaledBetaPose;
                if (jacobians != nullptr) {
                    const Eigen::MatrixXd& L = posePrior.prec_cho[compIdx];
                    // precision = L L^T
                    for (int i = 0; i < nSmplJoints; ++i) {
                        if (jacobians[i] != nullptr) {
                            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> > J(jacobians[i], nResids, 4);
                            J.topLeftCorner<Eigen::Dynamic, 3>(nResids - 1, 3).noalias() = L.middleRows<3>(i * 3).transpose() *
                                0.707106781186548 * commonData.scaledBetaPose;
                            J.rightCols<1>().setZero();
                            J.bottomLeftCorner<1, 3>().setZero();
                        }
                    }
                }                
                return true;
            }
            AvatarEvaluationCommonData<AvatarCostFunctorCache>& commonData;
            const GaussianMixture& posePrior;
            const int nSmplJoints;
        };

        /** Ceres analytic derivative cost function for shape prior error
         *  (This is extremely simple, just the squared l2-norm of w!) */
        struct AvatarShapePriorCostFunctor : ceres::CostFunction {
            AvatarShapePriorCostFunctor(int num_shape_keys, double beta_shape) :
                numShapeKeys(num_shape_keys), betaShape(beta_shape) {
                    set_num_residuals(numShapeKeys); // 1 for each shape key
                    std::vector<int> * paramBlockSizes = mutable_parameter_block_sizes();
                    paramBlockSizes->push_back(numShapeKeys); // 1 for each shape key
                }

            bool Evaluate(double const* const* parameters,
                    double* residuals,
                    double** jacobians) const final {
                Eigen::Map<Eigen::VectorXd> resid(residuals, numShapeKeys);
                Eigen::Map<const Eigen::VectorXd> w(parameters[0], numShapeKeys);
                resid.noalias() = w * betaShape;
                if (jacobians != nullptr) {
                    if (jacobians[0] != nullptr) {
                        Eigen::Map<Eigen::MatrixXd> J(jacobians[0], numShapeKeys, numShapeKeys);
                        J.noalias() = Eigen::MatrixXd::Identity(numShapeKeys, numShapeKeys) * betaShape;
                    }
                }
                return true;
            }
            const int numShapeKeys;
            const double betaShape;
        };

#ifdef TEST_COMPARE_AUTO_DIFF
        /** Auto diff cost function w/ derivative for Ceres
         *  (Extremely poorly optimized, used for checking correctness of analytic derivative) */
        struct AvatarICPAutoDiffCostFunctor {
            AvatarICPAutoDiffCostFunctor(AvatarEvaluationCommonData<AvatarCostFunctorCache> & common_data,
                    size_t cache_id, const CloudType& data_cloud, int data_point_id)
                : commonData(common_data), cacheId(cache_id), dataCloud(data_cloud), dataPointId(data_point_id) {

                    pointId = commonData.caches[cacheId].pointId;
                }
            template<class T>
                bool operator()(T const* const* params, T* residual) const {
                    using VecMap = Eigen::Map<Eigen::Matrix<T, 3, 1> >;
                    using ConstVecMap = Eigen::Map<const Eigen::Matrix<T, 3, 1> >;
                    using ConstQuatMap = Eigen::Map<const Eigen::Quaternion<T> >;

                    Eigen::Matrix<T, 3, Eigen::Dynamic> cloud(3, commonData.ava.model.numPoints());
                    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > cloudVec(cloud.data(), cloud.rows() * cloud.cols());

                    if (commonData.shape_enabled) {
                        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1> > wMap(params[commonData.ava.model.numJoints() + 1], commonData.ava.model.numShapeKeys(), 1);
                        cloudVec.noalias() = commonData.ava.model.keyClouds.cast<T>() * wMap + commonData.ava.model.baseCloud;
                    } else {
                        cloudVec.noalias() = commonData.ava.model.keyClouds.cast<T>() * commonData.ava.w.cast<T>() + commonData.ava.model.baseCloud;
                    }

                    Eigen::Matrix<T, 3, Eigen::Dynamic> jointPos =
                        cloud * commonData.ava.model.jointRegressor.cast<T>();

                    if (commonData.ava.model.useJointShapeRegressor) {
                        jointPos.resize(3, commonData.ava.model.numJoints());
                        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > jointPosVec(jointPos.data(), 3 * commonData.ava.model.numJoints());

                        if (commonData.shape_enabled) {
                            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1> > wMap(params[commonData.ava.model.numJoints() + 1], commonData.ava.model.numShapeKeys(), 1);
                            jointPosVec.noalias() = commonData.ava.model.jointShapeRegBase.cast<T>() + commonData.ava.model.jointShapeReg.cast<T>() * wMap; 
                        } else {
                            jointPosVec.noalias() = commonData.ava.model.jointShapeRegBase.cast<T>() + commonData.ava.model.jointShapeReg.cast<T>() * commonData.ava.w.cast<T>();
                        }
                    } else {
                        jointPos.noalias() = cloud * commonData.ava.model.jointRegressor.cast<T>();
                    }

                    Eigen::Matrix<T, 3, 1> offset = jointPos.col(0);
                    cloud.colwise() -= offset;
                    jointPos.colwise() -= offset;

                    VecMap resid(residual);
                    resid.setZero();
                    ConstVecMap rootPos(params[0]);
                    for (auto& assign: commonData.ava.model.assignedJoints[pointId]) {
                        Eigen::Matrix<T, 3, 1> vec = (cloud.col(pointId) - jointPos.col(assign.second)).template cast<T>();
                        for (int p = assign.second; p != -1; p = commonData.ava.model.parent[p]) {
                            vec = ConstQuatMap(params[p+1]).toRotationMatrix() * vec;
                            // Do not add if root joint since we want to add the rootPos
                            // instead in autodiff case
                            if (p) vec.noalias() += jointPos.col(p) - jointPos.col(commonData.ava.model.parent[p]);
                        }
                        resid.noalias() += assign.first * vec;
                    }
                    resid.noalias() += rootPos;
                    resid.noalias() -= dataCloud.col(dataPointId);
                    return true;
                }

            const CloudType& dataCloud;
            int dataPointId, pointId;
            size_t cacheId;
            AvatarEvaluationCommonData<AvatarCostFunctorCache>& commonData;
        };
#endif // TEST_COMPARE_AUTO_DIFF

        typedef nanoflann::KDTreeEigenColMajorMatrixAdaptor<
            CloudType, 3, nanoflann::metric_L2_Simple> KdTree;
        void findNN(const CloudType & data_cloud,
                const Eigen::VectorXi& data_part_labels,
                const std::vector<Eigen::VectorXi>& data_part_indices,
                const CloudType & model_cloud,
                const Eigen::VectorXi& model_part_labels,
                const std::vector<Eigen::VectorXi>& model_part_indices,
                std::vector<CloudType>& model_part_clouds,
                std::vector<bool>& point_visible,
                std::vector<std::vector<int> > & correspondences,
                std::vector<std::unique_ptr<KdTree>>& part_kd,
                int nn_step,
                bool invert = false) {

            if (invert) {
                size_t index; double dist;
                nanoflann::KNNResultSet<double> resultSet(1);
                // match each data point to a model point
                typedef nanoflann::KDTreeEigenColMajorMatrixAdaptor<
                    CloudType, 3, nanoflann::metric_L2_Simple> KdTree;
                std::vector<std::unique_ptr<KdTree>> modelPartKD;
                const size_t numParts = model_part_indices.size();
                for (size_t i = 0; i < numParts; ++i) {
                    const auto& indices = model_part_indices[i];
                    auto& partCloud = model_part_clouds[i];
                    for (int j = 0; j < indices.rows(); ++j) {
                        int k = indices[j];
                        partCloud.col(j).noalias() = model_cloud.col(k);
                    }
                    modelPartKD.emplace_back(new KdTree(model_part_clouds[i], 10));
                    modelPartKD.back()->index->buildIndex();
                }

                correspondences.resize(model_cloud.cols());
                for (int i = 0; i < model_cloud.cols(); ++i) {
                    correspondences[i].clear();
                }
                for (int i = 0; i < data_cloud.cols(); ++i) {
                    resultSet.init(&index, &dist);
                    const int partId = data_part_labels[i];
                    modelPartKD[partId]
                        ->index->findNeighbors(
                                resultSet,
                                data_cloud.data() + i * 3,
                                nanoflann::SearchParams(10));
                    correspondences[
                        model_part_indices[partId][index]
                    ].push_back(i);
                }
                // const int MAX_CORRES_PER_POINT = 1;
                // for (int i = 0; i < model_cloud.cols(); ++i) {
                //     if (correspondences[i].size() > MAX_CORRES_PER_POINT) {
                //         //std::cerr << correspondences[i].size() << "SZ\n";
                //         for (int j = 0; j < MAX_CORRES_PER_POINT; ++j) {
                //             int r = random_util::randint<int>(j, correspondences[i].size()-1);
                //             std::swap(correspondences[i][j], correspondences[i][r]);
                //         }
                //         correspondences[i].resize(MAX_CORRES_PER_POINT);
                //     }
                // }

            } else {
                size_t index; double dist;
                nanoflann::KNNResultSet<double> resultSet(1);

                // match each model point to a data point
                correspondences.resize(model_cloud.cols());
                for (int i = 0; i < model_cloud.cols(); ++i) {
                    correspondences[i].clear();
                }
                Eigen::VectorXi perPart(part_kd.size());
                perPart.setZero();
                for (int i = 0; i < model_cloud.cols(); i += nn_step) {
                    if (!point_visible[i]) continue;
                    resultSet.init(&index, &dist);
                    int partId = model_part_labels[i];
                    auto* kd_tree = part_kd[partId].get();
                    if (kd_tree) {
                        kd_tree->index->findNeighbors(resultSet, model_cloud.data() + i * 3, nanoflann::SearchParams(10));
                        correspondences[i].push_back(data_part_indices[partId][index]);
                        ++perPart[partId];
                    }
                }
                for (int i = 0; i < perPart.rows(); ++i){
                    std::cout << perPart(i) << " ";
                }
                std::cout << "!!\n";


                // if (ownTree) {
                //     delete kd_tree;
                // }
                /*
                   const int MAX_CORRES_PER_POINT = 200;
                   for (int i = 0; i < model_cloud.cols(); ++i) {
                   if (correspondences[i].size() > MAX_CORRES_PER_POINT) {
                //std::cerr << correspondences[i].size() << "SZ\n";
                for (int j = 0; j < MAX_CORRES_PER_POINT; ++j) {
                int r = random_util::randint<int>(j, correspondences[i].size()-1);
                std::swap(correspondences[i][j], correspondences[i][r]); 
                }
                correspondences[i].resize(MAX_CORRES_PER_POINT);
                }
                }
                */

            }
        }

#ifdef PCL_DEBUG_VISUALIZE
        void debugVisualize(const pcl::visualization::PCLVisualizer::Ptr& viewer,
                const CloudType& data_cloud, std::vector<std::vector<int> > correspondences,
               const std::vector<bool>& point_visible, AvatarEvaluationCommonData<AvatarCostFunctorCache>& common) {
            auto modelPclCloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
            auto dataPclCloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
            // auto matchedModelPointsCloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
            modelPclCloud->reserve(common.ava.cloud.cols());
            for (int i = 0 ; i < common.ava.cloud.cols(); ++i) {
                pcl::PointXYZRGBA pt;
                pt.getVector3fMap() = common.ava.cloud.col(i).cast<float>();
                if (!point_visible[i]) {
                    pt.r = pt.g = pt.b = 100;
                }  else {
                    pt.r = 255;
                    pt.g = 0;
                    pt.b = 0;
                }
                pt.a = 255;
                modelPclCloud->push_back(std::move(pt));
            }
            dataPclCloud->reserve(data_cloud.cols());
            for (int i = 0 ; i < data_cloud.cols(); ++i) {
                pcl::PointXYZRGBA pt;
                pt.getVector3fMap() = data_cloud.col(i).cast<float>();
                pt.r = 100;
                pt.g = 100;
                pt.b = 100;
                pt.a = 200;
                dataPclCloud->push_back(std::move(pt));
            }
            // matchedModelPointsCloud->reserve(common.caches.size());
            // common.PrepareForEvaluation(true, true);
            // for (auto& cache : common.caches) {
            //     cache.updateData(false);
            //     pcl::PointXYZRGBA pt;
            //     pt.getVector3fMap() = cache.resid.cast<float>();
            //     pt.r = 0;
            //     pt.g = 255;
            //     pt.b = 0;
            //     pt.a = 255;
            //     matchedModelPointsCloud->push_back(std::move(pt));
            // }

            viewer->setBackgroundColor(0, 0, 0);
            viewer->removePointCloud("cloud");
            viewer->removePointCloud("cloudData");
            viewer->removePointCloud("cachesCloud");
            viewer->removeAllShapes();
            viewer->addPointCloud(modelPclCloud, "cloud", 0);
            viewer->addPointCloud(dataPclCloud, "cloudData", 0);
            // viewer->addPointCloud(matchedModelPointsCloud, "cachesCloud", 0);
            /*
            for (int i = 0; i < common.ava.model.numJoints(); ++i) {
                pcl::PointXYZRGBA curr;
                curr.x = common.jointPosInit(0, i);
                curr.y = common.jointPosInit(1, i);
                curr.z = common.jointPosInit(2, i);
                //std::cerr << "Joint:" << joints[i]->name << ":" << curr.x << "," << curr.y << "," << curr.z << "\n";
                cv::Vec3f colorf(0.f, 0.f, 1.0f);
                std::string jointName = "avatarJoint" + std::to_string(i);
                viewer->removeShape(jointName, 0);
                viewer->addSphere(curr, 0.02, colorf[0], colorf[1], colorf[2], jointName, 0);
                // if (joints[i]->parent) {
                //     p parent = util::toPCLPoint(joints[i]->parent->posTransformed);
                //     std::string boneName = pcl_prefix + "avatarBone" + std::to_string(i);
                //     viewer->removeShape(boneName, viewport);
                //     viewer->addLine(curr, parent, colorf[2], colorf[1], colorf[0], boneName, viewport);
                // }
            }
            */

            for (size_t i = 0; i < correspondences.size(); ++i) {
                for (size_t j = 0; j < correspondences[i].size(); ++j) {
                    // if (random_util::uniform(0.0, 1.0) > 0.05) continue;
                    pcl::PointXYZ p1, p2;
                    p1.getVector3fMap() = common.ava.cloud.col(i).cast<float>();
                    p2.getVector3fMap() = data_cloud.col(correspondences[i][j]).cast<float>();
                    std::string name = "nn_line_" + std::to_string(i) +"_" + std::to_string(correspondences[i][j]);
                    viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(p2, p1, 0.0, 1.0, 0.0, name, 0);
                }
            }

            viewer->spin();
        }
#endif

#ifdef TEST_COMPARE_AUTO_DIFF
        /** Given a model point-data point pair, this function checks that autodiff gives the same result
         *  as the user-supplied analytic derivatives. Useful for verifying correctness. */
        void testCompareAutoDiff(AvatarOptimizer& opt, const ark::CloudType& data_cloud, int model_point_id, int data_point_id) {
            using namespace ceres;
            std::cerr << "INFO: COMPARING AUTO DIFF\n";

            std::cerr << data_cloud.col(data_point_id).transpose() << " DATA POINT\n";

            AvatarEvaluation1ommonData<AvatarCostFunctorCache> common(opt, true);
            common.scaledBetaShape = opt.betaShape;
            common.scaledBetaPose = opt.betaPose;

            std::cerr << common.shapedCloud.col(model_point_id).transpose() << " MODEL POINT, init\n";
            for (auto& ances : common.ancestor[model_point_id]) {
                std::cerr << ances.jid << ", p() = " << opt.ava.model.parent[ances.jid] << "\n";
                std::cerr << common.jointPosInit.col(ances.jid).transpose() << " t\n";
                std::cerr << opt.r[ances.jid].w() << ": " << opt.r[ances.jid].vec().transpose() << " q\n";
                std::cerr << "\n";
            }
            std::cerr << "\n";

            for (auto&assign : opt.ava.model.assignedJoints[model_point_id]) {
                std::cerr << assign.first <<":" << assign.second << " ";
            }
            std::cerr << "\n";

            common.caches.emplace_back(common, model_point_id);
            std::cerr << "Created dummy caches\n";

            DynamicAutoDiffCostFunction<AvatarICPAutoDiffCostFunctor>* cost_function = new DynamicAutoDiffCostFunction<AvatarICPAutoDiffCostFunctor>(
                    new AvatarICPAutoDiffCostFunctor(common, 0, data_cloud, data_point_id));
            cost_function->AddParameterBlock(3);
            for (int k = 0; k < opt.ava.model.numJoints(); ++k) {
                cost_function->AddParameterBlock(4);
            }
            if (common.shape_enabled) {
                cost_function->AddParameterBlock(opt.ava.model.numShapeKeys());
            }
            cost_function->SetNumResiduals(3);

            AvatarICPCostFunctor* our_cost_function = new AvatarICPCostFunctor(common, 0, data_cloud, data_point_id);
                             // NULL, pointParams[i]);

            std::vector<double*> params, fullParams;
            params.reserve(common.ancestor[model_point_id].size() + 1);
            params.push_back(common.ava.p.data());
            for (auto& ances : common.ancestor[model_point_id]) {
                params.push_back(opt.r[ances.jid].coeffs().data());
            }

            fullParams.push_back(common.ava.p.data());
            for (int i = 0; i < opt.ava.model.numJoints(); ++i) {
                fullParams.push_back(opt.r[i].coeffs().data());
            }

            if (common.shape_enabled) {
                params.push_back(common.ava.w.data());
                fullParams.push_back(common.ava.w.data());
            }

            double resid[3];
            double ** jaco = new double*[params.size()];
            jaco[0] = new double[3*3];
            for (int i = 1; i < params.size()-1; ++i) {
                jaco[i] = new double[3*4];
            }
            jaco[params.size()-1] = new double[3*opt.ava.model.numShapeKeys()];
            common.PrepareForEvaluation(true, true);
            our_cost_function->Evaluate(&params[0], resid, jaco); 
            std::cerr << "Residual ours \n" << resid[0] << " " << resid[1] << " " << resid[2] << "\n";

            double** jaco2 = new double*[fullParams.size()];
            jaco2[0] = new double[3*3];
            for (int i = 1; i < fullParams.size()-1; ++i) {
                jaco2[i] = new double[3*4];
            }
            jaco2[fullParams.size()-1] = new double[3*opt.ava.model.numShapeKeys()];
            cost_function->Evaluate(&fullParams[0], resid, jaco2); 
            std::cerr << "Residual theirs \n" << resid[0] << " " << resid[1] << " " << resid[2] << "\n";

            std::cerr << "JACOBIAN ours \n";
            for (int i = 0; i < params.size(); ++i) {
                int cols = 3+(i>0);
                if (i+1 == params.size()) cols = opt.ava.model.numShapeKeys();
                std::cerr << "BLOCK " << i;
                if (i>0 && (i < params.size()-1 || !common.shape_enabled)) std::cerr << " -> " << common.ancestor[model_point_id][i-1].jid + 1;
                std::cerr << "\n";
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < cols; ++k) {
                        std::cerr << jaco[i][j * cols + k] << "\t";
                    }
                    std::cerr << "\n";
                }
                std::cerr << "\n";
                delete[] jaco[i];
            }
            delete[] jaco;
            std::cerr << "JACOBIAN theirs \n";
            for (int i = 0; i < fullParams.size(); ++i) {
                int cols = 3+(i>0);
                if (i+1 == fullParams.size()) cols = opt.ava.model.numShapeKeys();
                bool good = false;
                if (i && (i < fullParams.size() - 1 || !common.shape_enabled)) {
                    for (auto& ances : common.ancestor[model_point_id]) {
                        if (ances.jid == i-1) {
                            good = true;
                            break;
                        }
                    }
                } else good = true;
                if (good) {
                    std::cerr << "BLOCK " << i << "\n";
                    for (int j = 0; j < 3; ++j) {
                        for (int k = 0; k < cols; ++k) {
                            std::cerr << jaco2[i][j * cols + k] << "\t";
                        }
                        std::cerr << "\n";
                    }
                    std::cerr << "\n";
                }
                delete[] jaco2[i];
            }
            std::cerr << "\n";
            delete[] jaco2;
            delete our_cost_function;
            delete cost_function;
            std::exit(0);
        }
#endif // TEST_COMPARE_AUTO_DIFF
    }

    AvatarOptimizer::AvatarOptimizer(
            Avatar& ava, const CameraIntrin& intrin,
            const cv::Size& image_size,
            int num_parts, const int* part_map)
        : ava(ava), intrin(intrin), imageSize(image_size),
          numParts(num_parts), partMap(part_map) {
        r.resize(ava.model.numJoints());

        modelPartIndices.resize(numParts);
        modelPartLabelCounts.resize(numParts);
        modelPartClouds.resize(numParts);
        modelPartLabelCounts.setZero();
        for (size_t i = 0; i < ava.model.numPoints(); ++i) {
            int mainJointId = ava.model.assignedJoints[i][0].second;
            ++modelPartLabelCounts(partMap[mainJointId]);
        }
        for (int i = 0; i < numParts; ++i) {
            if (modelPartLabelCounts(i) == 0) continue;
            modelPartClouds[i].resize(3, modelPartLabelCounts(i));
            modelPartIndices[i].resize(modelPartLabelCounts(i));
        }

        modelPartLabelCounts.setZero();
        for (size_t i = 0; i < ava.model.numPoints(); ++i) {
            int mainJointId = ava.model.assignedJoints[i][0].second;
            int partId = partMap[mainJointId];
            modelPartIndices[partId][modelPartLabelCounts(partId)] = i;
            ++modelPartLabelCounts(partId);
        }
    }

    void AvatarOptimizer::optimize(const Eigen::Matrix<double, 3, Eigen::Dynamic>& data_cloud,
            const Eigen::VectorXi& data_part_labels,
            int icp_iters, int num_threads) {
        // Convert to quaternion
        for (int i = 0; i < ava.model.numJoints(); ++i) {
            Eigen::AngleAxisd aa;
            aa.fromRotationMatrix(ava.r[i]);
            r[i] = aa;
        }

        AvatarEvaluationCommonData<AvatarCostFunctorCache> common(*this, true);
        common.numThreads = num_threads;//boost::thread::hardware_concurrency();;
        std::vector<std::vector<int> > correspondences;

#ifdef PCL_DEBUG_VISUALIZE
        auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewport"));
        viewer->initCameraParameters();
#endif

#ifdef TEST_COMPARE_AUTO_DIFF
        testCompareAutoDiff(*this, data_cloud, 0, 0);
#endif

        // Create separate point cloud for each body part
        AvatarRenderer renderer(ava, intrin);
        std::vector<bool> pointVisible(ava.cloud.size());

        std::vector<CloudType> partClouds(numParts);
        std::vector<Eigen::VectorXi> partIndices(numParts);
        Eigen::VectorXi partLabelCounts(numParts);
        partLabelCounts.setZero();
        for (size_t i = 0; i < data_part_labels.rows(); ++i) {
            ++partLabelCounts(data_part_labels(i));
        }
        for (int i = 0; i < numParts; ++i) {
            if (partLabelCounts(i) == 0) continue;
            partClouds[i].resize(3, partLabelCounts(i));
            partIndices[i].resize(partLabelCounts(i));
        }

        partLabelCounts.setZero();
        for (size_t i = 0; i < data_part_labels.rows(); ++i) {
            int partId = data_part_labels(i);
            partClouds[partId].col(partLabelCounts(partId)) =
                data_cloud.col(i);
            partIndices[partId][partLabelCounts(partId)] = i;
            ++partLabelCounts(partId);
        }

        // Build KD tree for each body part
        std::vector<std::unique_ptr<KdTree>> partKD;
        for (int i = 0; i < numParts; ++i) {
            if (partLabelCounts(i) == 0) {
                partKD.emplace_back(nullptr);
                continue;
            }
            partKD.emplace_back(new KdTree(partClouds[i], 10));
            partKD.back()->index->buildIndex();
        }

        // Store labels for each model skin point
        Eigen::VectorXi modelPartLabels(ava.model.numPoints());
        for (size_t i = 0; i < ava.model.numPoints(); ++i) {
            int mainJointId = ava.model.assignedJoints[i][0].second;
            modelPartLabels[i] = partMap[mainJointId];
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;
        // options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
        // options.preconditioner_type = ceres::PreconditionerType::CLUSTER_JACOBI;
        //options.dogleg_type = ceres::DoglegType::SUBSPACE_DOGLEG;
        // options.initial_trust_region_radius = 1e4;
        options.minimizer_progress_to_stdout = false;
        options.logging_type = ceres::LoggingType::SILENT;
        options.minimizer_type = ceres::LINE_SEARCH;
        // options.use_approximate_eigenvalue_bfgs_scaling = true;
        //options.check_gradients = true;
        options.line_search_direction_type = ceres::BFGS;
        options.line_search_interpolation_type = ceres::CUBIC;
        // options.max_num_line_search_direction_restarts = 2;
        // options.max_num_line_search_step_size_iterations = 10;
        // options.line_search_type = ceres::ARMIJO;
        //options.max_linear_solver_iterations = num_subiter;
        options.max_num_iterations = maxItersPerICP;
        // options.num_threads = common.numThreads;
        options.function_tolerance = 1e-4;
        options.dense_linear_algebra_library_type = ceres::DenseLinearAlgebraLibraryType::LAPACK;

        options.evaluation_callback = &common;
        if (!enableOcclusion) {
            std::fill(pointVisible.begin(), pointVisible.end(), true);
        }

        for (int icp_iter = 0; icp_iter < icp_iters; ++icp_iter) {
            // Perform point cloud occlusion detection
            BEGIN_PROFILE;
            renderer.update();
            if (enableOcclusion) {
                std::fill(pointVisible.begin(), pointVisible.end(), false);
                cv::Mat facesMap = renderer.renderFaces(imageSize);
                const auto& faces = renderer.getOrderedFaces();
                for (int r = 0; r < facesMap.rows; ++r) {
                    auto* ptr = facesMap.ptr<int32_t>(r);
                    for (int c = 0; c < facesMap.cols; ++c) {
                        if (~ptr[c]) {
                            pointVisible[faces[ptr[c]].second[0]]
                                = pointVisible[faces[ptr[c]].second[1]]
                                = pointVisible[faces[ptr[c]].second[2]] = true;
                        }
                    }
                }
                PROFILE(>> Occlusion);
            }

            // Find correspondences
            findNN(data_cloud, data_part_labels, partIndices,
                    ava.cloud, modelPartLabels, modelPartIndices,
                    modelPartClouds,
                    pointVisible, correspondences, partKD, nnStep,
                    /*invert*/ true);
            PROFILE(>> NN Corresponences);
            
            using namespace ceres;
            Problem problem;

            auto fakeQuaternionLocalParam = new ceres::FakeQuaternionParameterization();
            problem.AddParameterBlock(ava.p.data(), 3);
            for (int i = 0; i < ava.model.numJoints(); ++i)  {
                problem.AddParameterBlock(r[i].coeffs().data(), ROT_SIZE, fakeQuaternionLocalParam);
            }
            if (common.shapeEnabled) {
                problem.AddParameterBlock(ava.w.data(),
                        ava.model.numShapeKeys());
            }
            common.caches.clear();
            //std::vector<std::tuple<ceres::ResidualBlockId, int, int> > residuals;
            std::vector<std::vector<double*> > pointParams(ava.model.numPoints());
            size_t totalResiduals = 0;
            for (int i = 0; i < ava.model.numPoints(); ++i)  {
                if (correspondences[i].empty()) continue;
                totalResiduals += correspondences[i].size();
                auto& params = pointParams[i];
                common.caches.emplace_back(common, i);
                params.reserve(common.ancestor[i].size() + 1);
                params.push_back(ava.p.data());
                for (auto& ances : common.ancestor[i]) {
                    params.push_back(r[ances.jid].coeffs().data());
                }
                if (common.shapeEnabled) {
                    params.push_back(ava.w.data());
                }
            }

            /** Scale the function weights according to number of ICP type residuals. Otherwise the function terms become extremely imbalanced in some cases. */
            common.scaledBetaPose = betaPose * std::sqrt(totalResiduals) / 15.;
            common.scaledBetaShape = betaShape * std::sqrt(totalResiduals) / 15.;
            PROFILE(>> Construct problem: parameter blocks);
            // // DEBUG
            // std::vector<double*> params;
            // params.push_back(ava.p.data());
            // for (int k = 0; k < ava.model.numJoints(); ++k) {
            //     params.push_back(r[k].coeffs().data());
            // }
            // //END DEBUG
            int cid = 0;
            for (int i = 0; i < ava.model.numPoints(); ++i)  {
                if (correspondences[i].empty()) continue;
                for (int j : correspondences[i]) {
                    problem.AddResidualBlock(
                            new AvatarICPCostFunctor(common, cid, data_cloud, j),
                            NULL, pointParams[i]);
                }
                ++cid;
            }

            std::vector<double*> posePriorParams;
            posePriorParams.reserve(ava.model.numJoints() - 1);
            for (int i = 1; i < ava.model.numJoints(); ++i) {
                posePriorParams.push_back(r[i].coeffs().data());
            }
            if (betaPose > 0.) {
                problem.AddResidualBlock(new AvatarPosePriorCostFunctor(common), NULL, posePriorParams);
            }
            if (betaShape > 0.) {
                problem.AddResidualBlock(new AvatarShapePriorCostFunctor(ava.model.numShapeKeys(), common.scaledBetaShape), NULL, ava.w.data());
            }
            PROFILE(>> Construct problem: residual blocks);

#ifdef PCL_DEBUG_VISUALIZE
            debugVisualize(viewer, data_cloud, correspondences, pointVisible, common);
#endif

            // Run solver
            Solver::Summary summary;
            // PROFILE(>> Render in PCL);

            ceres::Solve(options, &problem, &summary);

            PROFILE(>> Solve);

            // output (for debugging)
            // std::cout << summary.FullReport() << "\n";

            // Convert from quaternion
            for (int i = 0; i < ava.model.numJoints(); ++i) {
                ava.r[i].noalias() = r[i].toRotationMatrix();
            }
            ava.update();
            PROFILE(>> Finish);
            // std::cout << ava.w.transpose() << "\n";

            /*
            // This block shows the value of each residual
            for (auto& res_tup : residuals) {
                double val;
                ceres::Problem::EvaluateOptions eo;
                eo.residual_blocks.push_back(std::get<0>(res_tup));
                problem.Evaluate(eo, &val, NULL, NULL, NULL);
                std::cout << val << " energy = ";
                std::cout << (ava.cloud.col(std::get<1>(res_tup)) - data_cloud.col(std::get<2>(res_tup))).squaredNorm() * 0.5 << "\n";
            }*/
        }
#ifdef PCL_DEBUG_VISUALIZE
        viewer->spin();
        viewer->close();
#endif
    }
}
