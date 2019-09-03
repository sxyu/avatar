#include "AvatarOptimizer.h"

#include <vector>
#include <Eigen/StdVector>
#include <ceres/ceres.h>
#include <nanoflann.hpp>
#include <boost/thread.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include "AvatarPCL.h"

#include "Avatar.h"
#include "Util.h"

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
            bool shape_enabled;

            AvatarEvaluationCommonData(AvatarOptimizer& opt, bool shape_enabled = false)
                : opt(opt), ava(opt.ava), nJoints(opt.ava.model.numJoints()), shape_enabled(shape_enabled) {
                _L.resize(nJoints * AvatarOptimizer::ROT_SIZE);
                ++nJoints;
                _R.resize(nJoints * nJoints);
                _t.resize(_R.size());
                shapedCloud.resize(3, ava.model.numPoints());

                if (!shape_enabled) CalcShape();
                else shape_computed = false;

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
            }

            /** Compute joint and point positions after applying shape keys*/
            void CalcShape() {
                Eigen::Map<Eigen::VectorXd> shapedCloudVec(shapedCloud.data(), 3 * shapedCloud.cols());

                /** Apply shape keys */
                shapedCloudVec.noalias() = ava.model.keyClouds * ava.w + ava.model.baseCloud; 

                /** Apply joint regressor */
                // TODO: use dense joint regressor with compressed cloud
                jointPosInit.noalias() = shapedCloud * ava.model.jointRegressor;

                /** Ensure root is at origin*/
                Eigen::Vector3d offset = jointPosInit.col(0);
                shapedCloud.colwise() -= offset;
                jointPosInit.colwise() -= offset;

                /** Find relative positions of each joint */
                jointVecInit.noalias() = jointPosInit;
                for (int i = ava.model.numJoints() - 1; i >= 1; --i) {
                    jointVecInit.col(i).noalias() -= jointVecInit.col(ava.model.parent[i]);
                }
                shape_computed = true;
            }

            void PrepareForEvaluation(bool evaluate_jacobians,
                    bool new_evaluation_point) final {
                // std::cerr << "PREP " << evaluate_jacobians << ", " << new_evaluation_point << "\n";
                if (new_evaluation_point) {
                    if (shape_enabled || !shape_computed) CalcShape();
                    jointVecInit.col(0).noalias() = ava.p;

                    R(-1, -1).setIdentity(); // -1 means 'to global'
                    t(-1, -1).setZero();//.noalias() = ava.p;
                    for (int i = 0; i < nJoints - 1; ++i) {
                        R(i, i).setIdentity();
                        Eigen::Quaterniond q = opt.r[i].template cast<double>();
                        Eigen::Matrix3d rot = q.toRotationMatrix();
                        t(i, i).setZero(); //.noalias() = jointVecInit.col(i);
                        int p = ava.model.parent[i];
                        for (int j = p; ; j = ava.model.parent[j]) {
                            R(j, i).noalias() = R(j, p) * rot;
                            t(j, i).noalias() = R(j, p) * jointVecInit.col(i) + t(j, p);
                            if (j == -1) break;
                        }

                        if (evaluate_jacobians) {
                            // Fill the derivatives of the local to parent local
                            // rotation matrix
                            q.coeffs() *= 2;
                            // Using rotation matrix from Terzakis et al.
                            // L(i, 3) <<  q.w(), -q.z(),  q.y(),
                            //             q.z(),  q.w(), -q.x(),
                            //            -q.y(),  q.x(),  q.w();
                            // L(i, 0) <<  q.x(),  q.y(),  q.z(),
                            //             q.y(), -q.x(), -q.w(),
                            //             q.z(),  q.w(), -q.x();
                            // L(i, 1) << -q.y(),  q.x(),  q.w(),
                            //             q.x(),  q.y(),  q.z(),
                            //            -q.w(),  q.z(), -q.y();
                            // L(i, 2) << -q.z(), -q.w(),  q.x(),
                            //             q.w(), -q.z(),  q.y(),
                            //             q.x(),  q.y(),  q.y();
                            
                            // Using Eigen's toRotationMatrix
                            double tqx = -2.*q.x(), tqy = -2.*q.y(), tqz = -2.*q.z();
                            L(i, 3) <<     0., -q.z(),  q.y(),
                                        q.z(),     0., -q.x(),
                                       -q.y(),  q.x(),     0.;
                            L(i, 0) <<     0.,  q.y(),  q.z(),
                                        q.y(),    tqx, -q.w(),
                                        q.z(),  q.w(),    tqx;
                            L(i, 1) <<    tqy,  q.x(),  q.w(),
                                        q.x(),     0.,  q.z(),
                                       -q.w(),  q.z(),    tqy;
                            L(i, 2) <<    tqz, -q.w(),  q.x(),
                                        q.w(),    tqz,  q.y(),
                                        q.x(),  q.y(),     0.;

                            /** Left-multiply by joint's global rotation */
                            L(i, 0) = R(-1, p) * L(i, 0);
                            L(i, 1) = R(-1, p) * L(i, 1);
                            L(i, 2) = R(-1, p) * L(i, 2);
                            L(i, 3) = R(-1, p) * L(i, 3);
                        }
                    }

                    for (auto& cache : caches) {
                        cache.updateData(evaluate_jacobians);
                    }
                }
            }
            /** Joint-to-ancestor joint rotation (j_ances=-1 is global) */
            inline Eigen::Matrix3d& R(int j_ancestor, int j) {
                return _R[nJoints * (j_ancestor+1) + j+1];
            }
            /** Joint-to-ancestor joint relative position (j_ances=-1 is global) */
            inline Eigen::Vector3d& t(int j_ancestor, int j) {
                return _t[nJoints * (j_ancestor+1) + j+1];
            }
            /** Combined left-side matrix to multiply into Jacobians for joint j, component t */
            inline Eigen::Matrix3d& L(int j, int t) {
                return _L[j * AvatarOptimizer::ROT_SIZE + t];
            }
            AvatarOptimizer& opt;
            Avatar& ava;
            int nJoints;

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
            
            /** Deduped, topo sorted ancestor joints for each skin point,
             *  combining all joint assignments for the point */
            std::vector<std::vector<Ancestor> > ancestor;

            /** Joint initial relative/absolute positions */
            CloudType jointVecInit, jointPosInit;

            /** baseCloud after applying shape keys (3 * num points) */
            CloudType shapedCloud;

            /** List of point-specific caches */
            std::vector<Cache> caches;

        private:
            /** Joint-to-ancestor joint rotation (j_ances=-1 is global) */
            std::vector<Eigen::Matrix3d, MatAlloc> _R;

            /** Joint-to-ancestor joint relative position (j_ances=-1 is global) */
            std::vector<Eigen::Vector3d, VecAlloc> _t;

            /** Combined left-side matrix to use for Jacobians for joint j, component t */
            std::vector<Eigen::Matrix3d, MatAlloc> _L;

            /** True if CalcShape() has been called, to avoid further calls */
            bool shape_computed;
        };

        /** Common method for each model point */
        struct AvatarICPCostFunctorCache {
            AvatarICPCostFunctorCache(AvatarEvaluationCommonData<AvatarICPCostFunctorCache>& common_data,
                    int point_id, bool enable_shape = false)
                : commonData(common_data), opt(common_data.opt), ava(common_data.opt.ava),
                  pointId(point_id), enableShape(enable_shape) {
                  jacobian.resize(commonData.ancestor[pointId].size());
            }

            using JacobianType = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
            using JacobianAlloc = Eigen::aligned_allocator<JacobianType>;

            bool get(double* residuals, double** jacobians) const {
                Eigen::Map<Eigen::Vector3d> residualMap(residuals);
                residualMap.noalias() = resid;
                if (jacobians != nullptr) {
                    if (jacobians[0] != nullptr) {
                        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                        J.setIdentity();
                    }
                    for (size_t i = 0; i < commonData.ancestor[pointId].size(); ++i) {
                        if (jacobians[i+1] != nullptr) {
                            Eigen::Map<JacobianType> J(jacobians[i+1]);
                            J.noalias() = jacobian[i];
                        }
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

                    for (size_t i = 0; i < commonData.ancestor[pointId].size(); ++i) {
                        // Set derivative for each parent rotation
                        auto& ances = commonData.ancestor[pointId][i];
                        int j = ances.jid; // 'middle' joint we are differenting wrt
                        auto& J = jacobian[i];
                        for (int t = 0; t < 4; ++t) {
                            // 4 loops
                            J.col(t).setZero();
                            for (int assign = 0; assign < ances.num_assign; ++assign) {
                                // up to 4 inner loops
                                int k = ances.assign[assign]; // 'outer' joint assigned to the point
                                double weight = ances.weight[assign];
                                // if (j == 4 && t == 3) {
                                //     std::cerr << "! " << k << " assign, wt " << weight << " tt " << (pointPosInit - commonData.jointPosInit.col(k)).transpose()  <<"\n" << commonData.L(j, t) << " L\n\n" << commonData.R(j, k) << " R\n\n" << commonData.t(j, k).transpose() << "t\n--\n\n";
                                // }
                                J.col(t) += weight * (commonData.L(j, t) *
                                        (commonData.R(j, k) *
                                        (pointPosInit - commonData.jointPosInit.col(k)) + commonData.t(j, k)));
                                // 42 flops
                            }
                            // <=168 flops
                        }
                        // <=672 flops
                    }
                    // <=16,128 flops
                    // TODO: support shape key parameters here
                    // either that or use ADMM (since may be too costly)
                }
                // all residuals: <=110,476,800 flops hopefully less
            }

            Eigen::Vector3d resid;
            std::vector<JacobianType, JacobianAlloc> jacobian;

            Avatar& ava;
            AvatarOptimizer& opt;
            AvatarEvaluationCommonData<AvatarICPCostFunctorCache>& commonData;
            int pointId;
            bool enableShape;
        };

        /** Analytic derivative cost function w/ derivative for Ceres */
        struct AvatarICPCostFunctor : ceres::CostFunction {
            AvatarICPCostFunctor(AvatarEvaluationCommonData<AvatarICPCostFunctorCache> & common_data,
                    size_t cache_id, const CloudType& data_cloud, int data_point_id)
                : commonData(common_data), cacheId(cache_id), dataCloud(data_cloud), dataPointId(data_point_id) {
                set_num_residuals(3);
                auto& cache = common_data.caches[cache_id];

                std::vector<ceres::int32> * paramBlockSizes = mutable_parameter_block_sizes();
                paramBlockSizes->push_back(3); // Root position
                for (size_t i = 0; i < cache.commonData.ancestor[cache.pointId].size(); ++i) {
                    paramBlockSizes->push_back(4); // Add rotation block for each ancestor
                }
                if (cache.enableShape) paramBlockSizes->push_back(cache.ava.model.numShapeKeys()); // Shape key weights?
            }

            bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const final {
                if (!commonData.caches[cacheId].get(residuals, jacobians)) return false;
                Eigen::Map<Eigen::Vector3d> resid(residuals);
                resid -= dataCloud.col(dataPointId);
                return true;
            }
            const CloudType& dataCloud;
            int dataPointId;
            size_t cacheId;
            AvatarEvaluationCommonData<AvatarICPCostFunctorCache>& commonData;
        };

        /** Auto diff cost function w/ derivative for Ceres
         *  (Extremely poorly optimized, used for checking correctness of analytic derivative) */
        struct AvatarICPAutoDiffCostFunctor {
            AvatarICPAutoDiffCostFunctor(AvatarEvaluationCommonData<AvatarICPCostFunctorCache> & common_data,
                    size_t cache_id, const CloudType& data_cloud, int data_point_id)
                : commonData(common_data), cacheId(cache_id), dataCloud(data_cloud), dataPointId(data_point_id) {

                pointId = commonData.caches[cacheId].pointId;
            }
            template<class T>
            bool operator()(T const* const* params, T* residual) const {
                using VecMap = Eigen::Map<Eigen::Matrix<T, 3, 1> >;
                using ConstVecMap = Eigen::Map<const Eigen::Matrix<T, 3, 1> >;
                using ConstQuatMap = Eigen::Map<const Eigen::Quaternion<T> >;
                VecMap resid(residual);
                resid.setZero();
                ConstVecMap rootPos(params[0]);
                for (auto& assign: commonData.ava.model.assignedJoints[pointId]) {
                    Eigen::Matrix<T, 3, 1> vec = (commonData.shapedCloud.col(pointId) - commonData.jointPosInit.col(assign.second)).cast<T>();
                    for (int p = assign.second; p != -1; p = commonData.ava.model.parent[p]) {
                        vec = ConstQuatMap(params[p+1]).toRotationMatrix() * vec;
                        vec.noalias() += commonData.jointVecInit.col(p);
                    }
                    resid.noalias() += assign.first * vec;
                }
                resid.noalias() -= dataCloud.col(dataPointId);
                return true;
            }

            const CloudType& dataCloud;
            int dataPointId, pointId;
            size_t cacheId;
            AvatarEvaluationCommonData<AvatarICPCostFunctorCache>& commonData;
        };

        void findNN(const CloudType & dataCloud, const CloudType & modelCloud,
            std::vector<std::vector<int> > & correspondences, bool invert = true) {

            if (invert) {
                size_t index; double dist;
                nanoflann::KNNResultSet<double> resultSet(1);
                // match each data point to a model point
                typedef nanoflann::KDTreeEigenColMajorMatrixAdaptor<
                    CloudType, 3, nanoflann::metric_L2_Simple> KdTree;
                KdTree kd(modelCloud, 10);
                kd.index->buildIndex();

                correspondences.resize(modelCloud.cols());
                for (int i = 0; i < modelCloud.cols(); ++i) {
                    correspondences[i].clear();
                }
                for (int i = 0; i < dataCloud.cols(); ++i) {
                    resultSet.init(&index, &dist);
                    kd.index->findNeighbors(resultSet, dataCloud.data() + i * 3, nanoflann::SearchParams(10));
                    correspondences[index].push_back(i);
                }
                const int MAX_CORRES_PER_POINT = 1;
                for (int i = 0; i < modelCloud.cols(); ++i) {
                    if (correspondences[i].size() > MAX_CORRES_PER_POINT) {
                        //std::cerr << correspondences[i].size() << "SZ\n";
                        for (int j = 0; j < MAX_CORRES_PER_POINT; ++j) {
                            int r = random_util::randint<int>(j, correspondences[i].size()-1);
                            std::swap(correspondences[i][j], correspondences[i][r]); 
                        }
                        correspondences[i].resize(MAX_CORRES_PER_POINT);
                    }
                }

            } else {
                size_t index; double dist;
                nanoflann::KNNResultSet<double> resultSet(1);
                // match each data point to a model point
                typedef nanoflann::KDTreeEigenColMajorMatrixAdaptor<
                    CloudType, 3, nanoflann::metric_L2_Simple> KdTree;
                KdTree kd(dataCloud, 10);
                kd.index->buildIndex();

                correspondences.resize(modelCloud.cols());
                for (int i = 0; i < modelCloud.cols(); ++i) {
                    correspondences[i].clear();
                }
                for (int i = 0; i < modelCloud.cols(); ++i) {
                    resultSet.init(&index, &dist);
                    kd.index->findNeighbors(resultSet, modelCloud.data() + i * 3, nanoflann::SearchParams(10));
                    correspondences[i].push_back(index);
                }
                /*
                   const int MAX_CORRES_PER_POINT = 200;
                   for (int i = 0; i < modelCloud.cols(); ++i) {
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

        void debugVisualize(const pcl::visualization::PCLVisualizer::Ptr& viewer,
                const CloudType& data_cloud, std::vector<std::vector<int> > correspondences, AvatarEvaluationCommonData<AvatarICPCostFunctorCache>& common) {
            auto modelPclCloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
            auto dataPclCloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
            auto cachesPclCloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
            modelPclCloud->reserve(common.ava.cloud.cols());
            for (int i = 0 ; i < common.ava.cloud.cols(); ++i) {
                pcl::PointXYZRGBA pt;
                pt.getVector3fMap() = common.ava.cloud.col(i).cast<float>();
                pt.r = 255;
                pt.g = 255;
                pt.b = 0;
                pt.a = 100;
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
            cachesPclCloud->reserve(common.caches.size());
            common.PrepareForEvaluation(true, true);
            for (auto& cache : common.caches) {
                cache.updateData(false);
                pcl::PointXYZRGBA pt;
                pt.getVector3fMap() = cache.resid.cast<float>();
                pt.r = 0;
                pt.g = 255;
                pt.b = 0;
                pt.a = 255;
                cachesPclCloud->push_back(std::move(pt));
            }

            viewer->setBackgroundColor(0, 0, 0);
            viewer->removePointCloud("cloud");
            viewer->removePointCloud("cloudData");
            viewer->removePointCloud("cachesCloud");
            viewer->removeAllShapes();
            viewer->addPointCloud(modelPclCloud, "cloud", 0);
            viewer->addPointCloud(dataPclCloud, "cloudData", 0);
            viewer->addPointCloud(cachesPclCloud, "cachesCloud", 0);
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
                    if (random_util::uniform(0.0, 1.0) > 0.05) continue;
                    pcl::PointXYZ p1, p2;
                    p1.getVector3fMap() = common.ava.cloud.col(i).cast<float>();
                    p2.getVector3fMap() = data_cloud.col(correspondences[i][j]).cast<float>();
                    std::string name = "nn_line_" + std::to_string(i) +"_" + std::to_string(correspondences[i][j]);
                    viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(p2, p1, 1.0, 0.0, 0.0, name, 0);
                }
            }

            viewer->addCoordinateSystem(1.0);
            viewer->spin();
        }

        /** Given a model point-data point pair, this function checks that autodiff gives the same result
         *  as the user-supplied analytic derivatives. Useful for verifying correctness. */
        void testCompareAutoDiff(AvatarOptimizer& opt, const ark::CloudType& data_cloud, int model_point_id, int data_point_id) {
            using namespace ceres;
            std::cerr << "INFO: COMPARING AUTO DIFF\n";

            std::cerr << data_cloud.col(data_point_id).transpose() << " DATA POINT\n";

            AvatarEvaluationCommonData<AvatarICPCostFunctorCache> common(opt, false);
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

            common.caches.emplace_back(common, model_point_id, false);
            std::cerr << "Created dummy caches\n";

            DynamicAutoDiffCostFunction<AvatarICPAutoDiffCostFunctor>* cost_function = new DynamicAutoDiffCostFunction<AvatarICPAutoDiffCostFunctor>(
                    new AvatarICPAutoDiffCostFunctor(common, 0, data_cloud, data_point_id));
            cost_function->AddParameterBlock(3);
            for (int k = 0; k < opt.ava.model.numJoints(); ++k) {
                cost_function->AddParameterBlock(4);
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

            double resid[3];
            double ** jaco = new double*[params.size()];
            jaco[0] = new double[3*3];
            for (int i = 1; i < params.size(); ++i) {
                jaco[i] = new double[3*4];
            }
            common.PrepareForEvaluation(true, true);
            our_cost_function->Evaluate(&params[0], resid, jaco); 
            std::cerr << "Residual ours \n" << resid[0] << " " << resid[1] << " " << resid[2] << "\n";

            double** jaco2 = new double*[fullParams.size()];
            jaco2[0] = new double[3*3];
            for (int i = 1; i < fullParams.size(); ++i) {
                jaco2[i] = new double[3*4];
            }
            cost_function->Evaluate(&fullParams[0], resid, jaco2); 
            std::cerr << "Residual theirs \n" << resid[0] << " " << resid[1] << " " << resid[2] << "\n";

            std::cerr << "JACOBIAN ours \n";
            for (int i = 0; i < params.size(); ++i) {
                int cols = 3+(i>0);
                std::cerr << "BLOCK " << i;
                if (i>0) std::cerr << " -> " << common.ancestor[model_point_id][i-1].jid + 1;
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
                bool good = false;
                if (i) {
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
        }
    }

    AvatarOptimizer::AvatarOptimizer(Avatar& ava) : ava(ava) {
        r.resize(ava.model.numJoints());
    }

    void AvatarOptimizer::optimize(const Eigen::Matrix<double, 3, Eigen::Dynamic>& data_cloud, int icp_iters) {
        for (int i = 0; i < ava.model.numJoints(); ++i) {
            Eigen::AngleAxisd aa;
            aa.fromRotationMatrix(ava.r[i]);
            r[i] = aa;
        }
        AvatarEvaluationCommonData<AvatarICPCostFunctorCache> common(*this, false);
        std::vector<std::vector<int> > correspondences;

        auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewport"));
        viewer->initCameraParameters();

        // testCompareAutoDiff(*this, data_cloud, 0, 0);
        for (int icp_iter = 0; icp_iter < icp_iters; ++icp_iter) {
            findNN(data_cloud, ava.cloud, correspondences);
            
            using namespace ceres;
            Problem problem;

            problem.AddParameterBlock(ava.p.data(), 3);
            for (int i = 0; i < ava.model.numJoints(); ++i)  {
                problem.AddParameterBlock(r[i].coeffs().data(), ROT_SIZE, new EigenQuaternionParameterization());
            }
            common.caches.clear();
            int pointsCnt = 0;
            int residualsCnt = 0;
            //std::vector<std::tuple<ceres::ResidualBlockId, int, int> > residuals;
            size_t avgParams = 0;
            std::vector<std::vector<double*> > pointParams(ava.model.numPoints());
            for (int i = 0; i < ava.model.numPoints(); ++i)  {
                if (correspondences[i].empty()) continue;
                auto& params = pointParams[i];
                common.caches.emplace_back(common, i, false);
                params.reserve(common.ancestor[i].size() + 1);
                params.push_back(ava.p.data());
                for (auto& ances : common.ancestor[i]) {
                    params.push_back(r[ances.jid].coeffs().data());
                }
                ++pointsCnt;
                avgParams += params.size();
            }
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
                    // residuals.emplace_back(
                     problem.AddResidualBlock(
                            new AvatarICPCostFunctor(common, cid, data_cloud, j),
                            NULL, pointParams[i]);
                         // , i, j);
                     // DynamicAutoDiffCostFunction<AvatarICPAutoDiffCostFunctor>* cost_function = new DynamicAutoDiffCostFunction<AvatarICPAutoDiffCostFunctor>(
                     //              new AvatarICPAutoDiffCostFunctor(common, cid, data_cloud, j));
                     // cost_function->AddParameterBlock(3);
                     // for (int k = 0; k < ava.model.numJoints(); ++k) {
                     //     cost_function->AddParameterBlock(4);
                     // }
                     // cost_function->SetNumResiduals(3);
                     // problem.AddResidualBlock(cost_function, NULL, params);
                     ++residualsCnt;
                }
                ++cid;
            }

            debugVisualize(viewer, data_cloud, correspondences, common);
            std::cerr << "Residuals: " << residualsCnt << " Avg params: " << static_cast<double>(avgParams) / pointsCnt << "\n";

            Solver::Options options;
            options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
            options.trust_region_strategy_type = ceres::DOGLEG;
            //options.dogleg_type = ceres::DoglegType::SUBSPACE_DOGLEG;
            options.initial_trust_region_radius = 1e4;
            options.minimizer_progress_to_stdout = true;
            options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
            options.minimizer_type = ceres::TRUST_REGION;
            //options.check_gradients = true;
            //options.line_search_direction_type = ceres::LBFGS;
            //options.max_linear_solver_iterations = num_subiter;
            options.max_num_iterations = 10;
            options.num_threads = 4;//boost::thread::hardware_concurrency();
            options.function_tolerance = 1e-5;
            options.evaluation_callback = &common;

            // Run solver
            Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // output (for debugging)
            std::cout << summary.FullReport() << "\n";

            for (int i = 0; i < ava.model.numJoints(); ++i) {
                ava.r[i].noalias() = r[i].toRotationMatrix();
            }
            ava.update();

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
    }
}