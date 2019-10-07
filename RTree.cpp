#include "RTree.h"

#include <fstream>
#include <chrono>
#include <cstdio>
#include <random>
#include <deque>
#include <mutex>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <boost/filesystem.hpp>
#include <Eigen/StdVector>

#include "Util.h"
#include "AvatarRenderer.h"

#define BEGIN_PROFILE auto _start = std::chrono::high_resolution_clock::now()
#define PROFILE(x) do{printf("%s: %f ms\n", #x, std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _start).count()); _start = std::chrono::high_resolution_clock::now(); }while(false)

namespace {
    // Compute Shannon entropy of a distribution (must be normalized)
    inline float entropy(const ark::RTree::Distribution & distr) {
        // Can this be vectorized?
        float entropy = 0.f;
        for (int i = 0; i < distr.rows(); ++i) {
            if (distr[i] < 1e-10) continue;
            entropy -= distr[i] * std::log2(distr[i]);
        }
        return entropy;
    }

    // Get depth at point in depth image, or return BACKGROUND_DEPTH
    // if in the background OR out of bounds
    inline float getDepth(const cv::Mat& depth_image, const ark::RTree::Vec2i& point) {
        if (point.y() < 0 || point.x() < 0 ||
                point.y() >= depth_image.rows || point.x() >= depth_image.cols)
            return ark::RTree::BACKGROUND_DEPTH;
        float depth = depth_image.at<float>(point.y(), point.x());
        if (depth <= 0.0) return ark::RTree::BACKGROUND_DEPTH;
        return depth;
    }

    /** Get score of single sample given by a feature */
    inline float scoreByFeature(cv::Mat depth_image,
            const ark::RTree::Vec2i& pix,
            const ark::RTree::Vec2& u,
            const ark::RTree::Vec2& v) {
            float sampleDepth = 
                depth_image.at<float>(pix.y(), pix.x());
            // Add feature u,v and round
            Eigen::Vector2f ut = u / sampleDepth, vt = v / sampleDepth;
            ark::RTree::Vec2i uti, vti;
            uti << static_cast<int16_t>(std::round(ut.x())), static_cast<int16_t>(std::round(ut.y()));
            vti << static_cast<int16_t>(std::round(vt.x())), static_cast<int16_t>(std::round(vt.y()));
            uti += pix; vti += pix;
            
            return (getDepth(depth_image, uti) - getDepth(depth_image, vti));
        }
}

namespace ark {
    const float RTree::BACKGROUND_DEPTH = 20.f;

    RTree::RNode::RNode() : leafid(-1) {};
    RTree::RNode::RNode(const Vec2& u, const Vec2& v, float thresh) :
                u(u), v(v), thresh(thresh), leafid(-1) {}

    enum {
        DATA_DEPTH,
        DATA_PART_MASK,
        _DATA_TYPE_COUNT
    };
    const int IMREAD_FLAGS[2] = { cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH, cv::IMREAD_GRAYSCALE };

    struct Sample {
        Sample () {}
        Sample(int index, const RTree::Vec2i& pix) : index(index), pix(pix) {};
        Sample(int index, int r, int c) : index(index), pix(c, r) {};

        // Image index
        int index;
        // Pixel position
        RTree::Vec2i pix; 
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    }; 
    using SampleVec = std::vector<Sample, Eigen::aligned_allocator<Sample> >;

    template<class DataSource>
    /** Responsible for handling data loading from abstract data source
     *  Interface: dataLoader.get(sample): get (depth, part mask) images for a sample
     *             dataLoader.preload(samples, a, b, numThreads): preload images for samples if possible */
    struct DataLoader {
        DataSource& dataSource;
        DataLoader(DataSource& data_source, int max_images_loaded) : dataSource(data_source), maxImagesLoaded(max_images_loaded) {}

        /** Precondition: samples must be sorted by image index on [a, ..., b-1] */
        bool preload(const SampleVec& samples, int a, int b, int numThreads) {
            std::vector<int> images;
            images.push_back(samples[a].index);
            for (int i = a + 1; i < b; ++i) {
                if (samples[i].index != images.back()) {
                    images.push_back(samples[i].index);
                }
            }
            images.resize(std::unique(images.begin(), images.end()) - images.begin());
            if (images.size() > maxImagesLoaded) {
                std::cerr << "INFO: Number of images too large (" << images.size() <<
                    " > " << maxImagesLoaded << "), not preloading\n";
                return false;
            }
            
            imageIdx.resize(dataSource.size(), -1);

            size_t loadSize = data.size();
            for (int im : images) {
                if (imageIdx[im] == -1) {
                    loadSize += 1;
                }
            }
            if (loadSize > maxImagesLoaded) {
                clear();
            } else {
                std::vector<int> newImages;
                for (int im : images) {
                    if (imageIdx[im] == -1) {
                        newImages.push_back(im);
                    }
                }
                images.swap(newImages);
            }
            if (images.empty()) return true;

            std::vector<std::thread> threads;
            std::mutex mtx;
            size_t i = 0, basei = data.size();
            data.resize(basei + images.size());
            revImageIdx.resize(basei + images.size());;

            auto worker = [&] {
                size_t thread_i;
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        if (i >= images.size()) break;
                        thread_i = i++;
                    }
                    data[basei + thread_i] = dataSource.load(images[thread_i]);
                    imageIdx[images[thread_i]] = basei + thread_i;
                    revImageIdx[basei + thread_i] = images[thread_i];
                }
            };
            for (int i = 0; i < numThreads; ++i) {
                threads.emplace_back(worker);
            }
            for (int i = 0; i < numThreads; ++i) {
                threads[i].join();
            }
            return true;
        }

        const std::array<cv::Mat, 2>& get(const Sample& sample) const {
            int iidx = sample.index >= imageIdx.size() ? -1 : imageIdx[sample.index];
            if (iidx < 0) {
                cv::Mat im0, im1;
                return dataSource.load(sample.index);
            }
            return data[iidx];
        }

        void clear() {
            data.clear();
            for (int i : revImageIdx) {
                imageIdx[i] = -1;
            }
            revImageIdx.clear();
        }

        std::vector<std::array<cv::Mat, 2> > data;
        std::vector<int> imageIdx, revImageIdx;
        int maxImagesLoaded;
    };

    /** Internal trainer implementation */
    template<class DataSource>
    class Trainer {
    public:
        struct Feature {
            RTree::Vec2 u, v;
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };

        Trainer() =delete;
        Trainer(const Trainer&) =delete;
        Trainer(Trainer&&) =delete;

        Trainer(std::vector<RTree::RNode>& nodes,
                std::vector<RTree::Distribution>& leaf_data,
                DataSource& data_source,
                int num_parts,
                int max_images_loaded)
            : nodes(nodes), leafData(leaf_data),
              dataLoader(data_source, max_images_loaded), numParts(num_parts) {
        }

        void train(int num_images, int num_points_per_image, int num_features,
                   int max_probe_offset, int min_samples, int max_tree_depth,
                   int samples_per_feature, int threshes_per_feature, int num_threads,
                   bool verbose, bool skip_init = false, bool skip_train = false) {
            this->verbose = verbose;

            if (!skip_init) {
                // Initialize new samples
                initTraining(num_images, num_points_per_image, max_tree_depth);
            }

            if (skip_train) return;
            
            if (verbose) {
                std::cerr << "Init RTree training with maximum depth " << max_tree_depth << "\n";
            }

            // Train
            numFeatures = num_features;
            maxProbeOffset = max_probe_offset;
            minSamples = min_samples;
            numThreads = num_threads;
            numImages = num_images;
            samplesPerFeature = samples_per_feature;
            threshesPerFeature = threshes_per_feature;
            nodes.resize(1);
            trainFromNode(nodes[0], 0, static_cast<int>(samples.size()), max_tree_depth);
            if (verbose) {
                std::cerr << "Training finished\n";
            }
        }

        void writeSamples(const std::string & path) {
            std::ofstream ofs(path, std::ios::out | std::ios::binary);
            dataLoader.dataSource.serialize(ofs);
            reorderByImage(0, samples.size());
            ofs.write("S\n", 2);
            size_t last_idx = 0;
            util::write_bin(ofs, samples.size());
            for (size_t i = 0; i <= samples.size(); ++i) {
                if (i == samples.size() ||
                    samples[i].index != samples[last_idx].index) {
                    util::write_bin(ofs, samples[last_idx].index);
                    util::write_bin(ofs, int(i - last_idx));
                    for (size_t j = last_idx; j < i; ++j) {
                        util::write_bin(ofs, samples[j].pix[0]);
                        util::write_bin(ofs, samples[j].pix[1]);
                    }
                    last_idx = i;
                }
            }
            util::write_bin(ofs, -1);
            util::write_bin(ofs, -1);
            ofs.close();
        }

        void readSamples(const std::string & path, bool verbose = false) {
            std::ifstream ifs(path, std::ios::in | std::ios::binary);
            if (verbose) {
                std::cout << "Recovering data source from samples file\n";
            }
            dataLoader.dataSource.deserialize(ifs);
            if (verbose) {
                std::cout << "Reading samples from samples file\n";
            }
            char marker[2];
            ifs.read(marker, 2);
            if (strncmp(marker, "S\n", 2)) {
                std::cerr << "ERROR: Invalid or corrupted samples file at " << path << "\n";
                return;
            }
            size_t numSamplesTotal;
            util::read_bin(ifs, numSamplesTotal);
            samples.reserve(numSamplesTotal);
            while (ifs) {
                int imgIndex, imgSamps;
                util::read_bin(ifs, imgIndex);
                util::read_bin(ifs, imgSamps);
                if (verbose && imgIndex % 20 == 0 && imgIndex >= 0) {
                    std::cout << "Reading samples for image #" << imgIndex << " with " << imgSamps << " sample pixels\n";
                }
                if (!ifs || imgSamps < 0) break;
                while (imgSamps--) {
                    samples.emplace_back();
                    Sample& sample = samples.back();
                    sample.index = imgIndex;
                    util::read_bin(ifs, sample.pix[0]);
                    util::read_bin(ifs, sample.pix[1]);
                }
            }
            ifs.close();
        }

    private:
        std::mutex trainMutex;
        void trainFromNode(RTree::RNode& node, int start, int end, uint32_t depth) {
            if (depth <= 1 || end - start <= minSamples) {
                // Leaf
                node.leafid = static_cast<int>(leafData.size());
                if (verbose) {
                    std::cerr << "Added leaf node: id=" << node.leafid << "\n";
                }
                leafData.emplace_back();
                leafData.back().resize(numParts);
                leafData.back().setZero();
                for (int i = start; i < end; ++i) {
                    auto samplePart = dataLoader.get(samples[i])[DATA_PART_MASK]
                        .template at<uint8_t>(samples[i].pix.y(), samples[i].pix.x());
                    leafData.back()(samplePart) += 1.f;
                }
                leafData.back() /= leafData.back().sum();
                return;
            } 
            if (verbose) {
                std::cerr << "Training internal node, remaining depth: " << depth <<
                    ". Current data interval: " << start << " to " << end << "\n";
            }

            using FeatureVec = std::vector<Feature, Eigen::aligned_allocator<Feature> >; 
            FeatureVec candidateFeatures;
            candidateFeatures.resize(numFeatures);

            static const double MIN_PROBE = 0.1;
            // Create random features
            for (auto& feature : candidateFeatures) {  
                // Create random feature in-place
                feature.u.x() =
                    random_util::uniform(-maxProbeOffset + MIN_PROBE, maxProbeOffset - MIN_PROBE);
                feature.u.x() += (feature.u.x() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                feature.u.y() = random_util::uniform(-maxProbeOffset + MIN_PROBE, maxProbeOffset - MIN_PROBE);
                feature.u.y() += (feature.u.y() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                feature.v.x() = random_util::uniform(-maxProbeOffset + MIN_PROBE, maxProbeOffset - MIN_PROBE);
                feature.v.x() += (feature.v.x() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
                feature.v.y() = random_util::uniform(-maxProbeOffset + MIN_PROBE, maxProbeOffset - MIN_PROBE);
                feature.v.y() += (feature.v.y() >= 0.0 ? MIN_PROBE : -MIN_PROBE);
            }

            // Precompute features scores
            if (verbose) {
                std::cerr << "Allocating memory and sampling...\n";
            }
            SampleVec subsamples;
            subsamples = random_util::choose(samples, start, end, samplesPerFeature);
            Eigen::MatrixXd sampleFeatureScores(subsamples.size(), numFeatures);
            Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> sampleParts(subsamples.size());
            dataLoader.preload(subsamples, 0, subsamples.size(), numThreads);

            std::vector<std::thread> threadMgr;

            if (verbose) {
                std::cerr << "Computing features on sparse samples...\n";
            }
            size_t sampleCount = 0;
            // This worker loads and computes the pixel scores and parts for sparse features
            auto sparseFeatureWorker = [&]() {
                int sampleId;
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(trainMutex);
                        if (sampleCount >= subsamples.size()) break;
                        sampleId = sampleCount++;
                    };
                    auto& sample = subsamples[sampleId];
                    const auto& dataArr = dataLoader.get(sample);

                    sampleParts(sampleId) = dataArr[DATA_PART_MASK]
                            .template at<uint8_t>(sample.pix[1], sample.pix[0]);

                    for (int featureId = 0; featureId < numFeatures; ++featureId) {
                        auto& feature = candidateFeatures[featureId];
                        sampleFeatureScores(sampleId, featureId) =
                            scoreByFeature(dataArr[DATA_DEPTH],
                                    sample.pix, feature.u, feature.v);
                    }
                }
            };

            for (int i = 0; i < numThreads; ++i) {
                threadMgr.emplace_back(sparseFeatureWorker);
            }
            for (int i = 0; i < numThreads; ++i) {
                threadMgr[i].join();
            }
            threadMgr.clear();

            if (verbose) {
                std::cerr << "Optimizing information gain on sparse samples...\n";
            }

            // Find best information gain
            std::vector<std::vector<std::array<float, 2> > > featureThreshes(numFeatures);

            int featureCount = numFeatures;
            // bestInfoGains.setConstant(-FLT_MAX);
            // Eigen::VectorXf bestInfoGains(numThreads, 1);
            // Eigen::VectorXf bestThreshs(numThreads, 1);
            // FeatureVec bestFeatures(numThreads);

            bool shortCircuitOnFeatureOpt = end-start == subsamples.size();

            // This worker finds threshesPerSample optimal thresholds for each feature on the selected sparse features
            auto threshWorker = [&](int thread_id) {
                // float& bestInfoGain = bestInfoGains(thread_id);
                // float& bestThresh = bestThreshs(thread_id);
                // Feature& bestFeature = bestFeatures[thread_id];
                int featureId;
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(trainMutex);
                        if (featureCount <= 0) break;

                        if (verbose && end-start > 50000 &&
                                featureCount % 100 == 0) {
                            std::cerr << " Sparse features to evaluate: " << featureCount << "\n";
                        }
                        featureId = --featureCount;
                    }

                    auto& feature = candidateFeatures[featureId];
                    float optimalThresh;
                    // float infoGain =
                    computeOptimalThreshes(
                            subsamples, sampleFeatureScores, sampleParts,
                            featureId, featureThreshes[featureId], shortCircuitOnFeatureOpt);

                    // if (infoGain >= bestInfoGain) {
                        // bestInfoGain = infoGain;
                        // bestThresh = optimalThresh;
                        // bestFeature = feature;
                    // }
                }
            };

            for (int i = 0; i < numThreads; ++i) {
                threadMgr.emplace_back(threshWorker, i);
            }
            for (int i = 0; i < numThreads; ++i) {
                threadMgr[i].join();
            }

            float bestThresh, bestInfoGain = -FLT_MAX;
            Feature bestFeature;

            if (shortCircuitOnFeatureOpt) {
                // Interval is very short (only has subsamples), reuse earlier computations
                for (size_t featureId = 0; featureId < numFeatures; ++featureId) {
                    auto& bestFeatureThresh = featureThreshes[featureId][0];
                    if (bestFeatureThresh[0] > bestInfoGain) {
                        bestInfoGain = bestFeatureThresh[0];
                        bestThresh = bestFeatureThresh[1];
                        bestFeature = candidateFeatures[featureId];
                    }
                }
            } else {
                // Interval is long
                if (verbose) {
                    std::cerr << "Computing part distributions for each candidate feature/threshold pair...\n";
                }
                sampleFeatureScores.resize(0, 0);
                sampleParts.resize(0);
                { 
                    SampleVec _;
                    subsamples.swap(_);
                }
                bool preloaded = dataLoader.preload(samples, start, end, numThreads);

                std::vector<Eigen::MatrixXi, Eigen::aligned_allocator<Eigen::MatrixXi> > featureThreshDist(numFeatures); 
                for (int i = 0; i < numFeatures; ++i) {
                    featureThreshDist[i].resize(featureThreshes[i].size(), numParts * 2);
                    featureThreshDist[i].setZero();
                }

                threadMgr.clear();
                sampleCount = start;
                // Compute part distributions for each feature/threshold pair
                auto featureDistributionWorker = [&]() {
                    int sampleId;
                    while (true) {
                        {
                            std::lock_guard<std::mutex> lock(trainMutex);
                            if (sampleCount >= end) break;
                            sampleId = sampleCount++;
                        };
                        auto& sample = samples[sampleId];
                        const auto& dataArr = dataLoader.get(sample);

                        uint32_t samplePart = dataArr[DATA_PART_MASK]
                            .template at<uint8_t>(sample.pix[1], sample.pix[0]);

                        for (int featureId = 0; featureId < numFeatures; ++featureId) {
                            auto& feature = candidateFeatures[featureId];
                            auto& distMat = featureThreshDist[featureId];
                            float score = scoreByFeature(dataArr[DATA_DEPTH],
                                        sample.pix, feature.u, feature.v);
                            for (size_t threshId = 0; threshId < featureThreshes[featureId].size(); ++threshId) {
                                uint32_t part = (score > featureThreshes[featureId][threshId][1]) ? samplePart : samplePart + numParts;
                                {
                                    std::lock_guard<std::mutex> lock(trainMutex);
                                    ++distMat(threshId, part);
                                }
                            }
                        }
                    }
                };

                for (int i = 0; i < numThreads; ++i) {
                    threadMgr.emplace_back(featureDistributionWorker);
                }
                for (int i = 0; i < numThreads; ++i) {
                    threadMgr[i].join();
                }
                if (verbose) {
                    std::cerr << "Finding optimal feature...\n";
                }
                threadMgr.clear();

                featureCount = numFeatures;
                auto featureOptWorker = [&]() {
                    // float& bestInfoGain = bestInfoGains(thread_id);
                    // float& bestThresh = bestThreshs(thread_id);
                    // Feature& bestFeature = bestFeatures[thread_id];
                    int featureId;
                    while (true) {
                        {
                            std::lock_guard<std::mutex> lock(trainMutex);
                            if (featureCount <= 0) break;

                            if (verbose && end-start > 50000 &&
                                    featureCount % 100 == 0) {
                                std::cerr << " Candidate features to evaluate: " << featureCount << "\n";
                            }
                            featureId = --featureCount;
                        }
                        auto& feature = candidateFeatures[featureId];
                        auto& distMat = featureThreshDist[featureId];

                        float featureBestInfoGain = -FLT_MAX;
                        float featureBestThresh = -1.;

                        for (size_t threshId = 0; threshId < featureThreshes[featureId].size(); ++threshId) {
                            auto distLeft = distMat.block<1, Eigen::Dynamic>(threshId, 0, 1, numParts);
                            auto distRight = distMat.block<1, Eigen::Dynamic>(threshId, numParts, 1, numParts);

                            float lsum = distLeft.sum();
                            float rsum = distRight.sum();

                            float leftEntropy = entropy(distLeft.template cast<float>() / lsum);
                            float rightEntropy = entropy(distRight.template cast<float>() / rsum);
                            // Compute the information gain
                            float infoGain = - lsum * leftEntropy + rsum * rightEntropy;
                            if (infoGain > 0) {
                                std::cerr << "FATAL: Possibly overflow detected during training, exiting. Internal data: left entropy "
                                    << leftEntropy << " right entropy "
                                    << rightEntropy << " information gain "
                                    << infoGain<< "\n";
                                std::exit(2);
                            }
                            if (infoGain > featureBestThresh) {
                                featureBestInfoGain = infoGain;
                                featureBestThresh = featureThreshes[featureId][threshId][1];
                            }
                        }
                        {
                            std::lock_guard<std::mutex> lock(trainMutex);
                            if (featureBestThresh > bestThresh) {
                                bestInfoGain = featureBestInfoGain;
                                bestFeature = feature;
                                bestThresh = featureBestThresh;
                            }
                        }
                    }
                };
                for (int i = 0; i < numThreads; ++i) {
                    threadMgr.emplace_back(featureOptWorker);
                }
                for (int i = 0; i < numThreads; ++i) {
                    threadMgr[i].join();
                }
            }

            int mid = split(start, end, bestFeature, bestThresh);

            if (verbose) {
                std::cerr << "> Best info gain " << bestInfoGain << ", thresh " << bestThresh << ", feature.u " << bestFeature.v.x() << "," << bestFeature.v.y() <<", features.v" << bestFeature.u.x() << "," << bestFeature.u.y() << "\n";
            }
            //std::cerr << bestThreadId << "BT thresh" << bestThreshs(bestThreadId) << ",u\n" << bestFeatures[bestThreadId].u << "\nv\n" <<bestFeatures[bestThreadId].v << "\n";
            if (mid == end || mid == start) {
                // force leaf
                trainFromNode(node, start, end, 0);
                /*
                std::cerr << "EVIL "<< start << " " << mid << " " << end << " \n";
                std::cerr << bestFeatures[bestThreadId].u << " U\n";
                std::cerr << bestFeatures[bestThreadId].v << " V\n";
                std::cerr << bestThreshs << " Threshs\n";
                std::cerr << bestInfoGains << " InfoGain\n";
                std::cerr << bestThreadId << " thead\n\n";

                for (int i = start; i < end; ++i) {
                    std::cerr << " " << 
                        scoreByFeature(
                                data[DATA_DEPTH][samples[i].index],
                                samples[i].pix,
                                bestFeatures[bestThreadId].u,
                                bestFeatures[bestThreadId].v);
                }
                std::cerr << "\n";
                std::exit(1);
                */
                return;
            }
            node.thresh = bestThresh;
            node.u = bestFeature.u;
            node.v = bestFeature.v;

            node.lnode = static_cast<int>(nodes.size());
            nodes.emplace_back();
            trainFromNode(nodes.back(), start, mid, depth - 1);

            node.rnode = static_cast<int>(nodes.size());
            nodes.emplace_back();
            trainFromNode(nodes.back(), mid, end, depth - 1);
        }

        // Compute information gain (mutual information scaled and shifted) by choosing optimal threshold
        // output into optimal_thresh.col(feature_id)
        // best <= threshesPerSample thresholds are found and returned in arbitrary order
        // if place_best_thresh_first then puts the absolute best threshold first, rest still arbitrary order
        void computeOptimalThreshes(
            const SampleVec& samples,
            const Eigen::MatrixXd& sample_feature_scores,
            const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>& sample_parts,
            int feature_id, std::vector<std::array<float, 2> >& optimal_threshes,
            bool place_best_thresh_first = false) {

            // Initially everything is in left set
            RTree::Distribution distLeft(numParts), distRight(numParts);
            distLeft.setZero();
            distRight.setZero();

            // Compute scores
            std::vector<std::pair<float, int> > samplesByScore;
            samplesByScore.reserve(samples.size());
            for (size_t i = 0; i < samples.size(); ++i) {
                const Sample& sample = samples[i];
                samplesByScore.emplace_back(
                        sample_feature_scores(i, feature_id), i);
                uint8_t samplePart = sample_parts(i);
                if (samplePart >= distLeft.size()) {
                    std::cerr << "FATAL: Invalid sample " << int(samplePart) << " detected during RTree training, "
                                 "please check the randomization code\n";
                    std::exit(0);
                }
                distLeft[samplePart] += 1.f;
            }
            static auto scoreComp = [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
                return a.first < b.first;
            };
            std::sort(samplesByScore.begin(), samplesByScore.end(), scoreComp);

            // Start everything in the left set ...
            float lastScore = -FLT_MAX;
            for (size_t i = 0; i < samplesByScore.size()-1; ++i) {
                // Update distributions for left, right sets
                int idx = samplesByScore[i].second;
                uint8_t samplePart = sample_parts(idx);
                distLeft[samplePart] -= 1.f;
                distRight[samplePart] += 1.f;
                if (lastScore == samplesByScore[i].first) continue;
                lastScore = samplesByScore[i].first;

                float left_entropy = entropy(distLeft / distLeft.sum());
                float right_entropy = entropy(distRight / distRight.sum());
                // Compute the information gain
                float infoGain = - ((samples.size() - i - 1) * left_entropy
                                 + (i+1)                     * right_entropy);
                if (infoGain > 0) {
                    std::cerr << "FATAL: Possibly overflow detected during training, exiting. Internal data: left entropy "
                        << left_entropy << " right entropy "
                        << right_entropy << " information gain "
                        << infoGain<< "\n";
                    std::exit(2);
                }
                // Add to candidate threshes
                optimal_threshes.push_back({infoGain, random_util::uniform(samplesByScore[i].first, samplesByScore[i+1].first)});
            }
            std::nth_element(optimal_threshes.begin(), optimal_threshes.begin() + threshesPerFeature,
                             optimal_threshes.end(), std::greater<std::array<float, 2> >());
            optimal_threshes.resize(threshesPerFeature);
            if (place_best_thresh_first) {
                std::nth_element(optimal_threshes.begin(), optimal_threshes.begin() + 1, optimal_threshes.end(), std::greater<std::array<float, 2> >());
            }
        }

        void initTraining(int num_images, int num_points_per_image, int max_tree_depth) {
            // 1. Choose num_images random images u.a.r. from given image list
            std::vector<int> allImages(dataLoader.dataSource.size());
            std::iota(allImages.begin(), allImages.end(), 0);
            chosenImages = allImages.size() > static_cast<size_t>(num_images) ?
                random_util::choose(allImages, num_images) : std::move(allImages);

            // 2. Choose num_points_per_image random foreground pixels from each image,
            for (size_t i = 0; i < chosenImages.size(); ++i) {
                if (verbose && i % 20 == 19) {
                    std::cerr << "Preprocessing data: " << i+1 << " of " << num_images << "\n";
                }
                cv::Mat mask = dataLoader.get(Sample(chosenImages[i], 0, 0))[DATA_PART_MASK];
                // cv::Mat mask2 = dataLoader.get(Sample(chosenImages[i], 0, 0))[DATA_PART_MASK];
                // cv::hconcat(mask, mask2, mask);
                // cv::resize(mask, mask, mask.size() / 2);
                // cv::imshow("MASKCat", mask);
                // cv::waitKey(0);
                std::vector<RTree::Vec2i, Eigen::aligned_allocator<RTree::Vec2i> > candidates;
                for (int r = 0; r < mask.rows; ++r) {
                    auto* ptr = mask.ptr<uint8_t>(r);
                    for (int c = 0; c < mask.cols; ++c) {
                        if (ptr[c] != 255) {
                            candidates.emplace_back();
                            candidates.back() << c, r;
                        }
                    }
                }
                std::vector<RTree::Vec2i, Eigen::aligned_allocator<RTree::Vec2i> > chosenCandidates =
                    (candidates.size() > static_cast<size_t>(num_points_per_image)) ?
                    random_util::choose(candidates, num_points_per_image) : std::move(candidates);
                for (auto& v : chosenCandidates) {
                    samples.emplace_back(chosenImages[i], v);
                }
            }

            // Note to future self: DO NOT SHUFFLE SAMPLES
            // Order of samples is important for caching.
            if(verbose) {
                std::cerr << "Preprocessing done, sparsely verifying data validity before training...\n";
            }
            for (int i = 0; i < samples.size(); i += std::max<int>(samples.size() / 100, 1)) {
                auto& sample = samples[i];
                cv::Mat mask = dataLoader.get(sample)[DATA_PART_MASK];
                if (mask.at<uint8_t>(sample.pix[1], sample.pix[0]) == 255) {
                    std::cerr << "FATAL: Invalid data detected during verification: background pixels were included in samples.\n";
                    std::exit(0);
                }
            }
            if(verbose) {
                std::cerr << "Result: data is valid\n";
            }
        }

        // Split samples {start ... end-1} by feature+thresh in-place and return the dividing index
        // left (less) set willInit  be {start ... idx-1}, right (greater) set is {idx ... end-1}  
        int split(int start, int end, const Feature& feature, float thresh) {
            int nextIndex = start;
            for (int i = start; i < end; ++i) {
                const Sample& sample = samples[i];
                if (scoreByFeature(dataLoader.get(sample)[DATA_DEPTH],
                            sample.pix, feature.u, feature.v) < thresh) {
                    if (nextIndex != i) {
                        std::swap(samples[nextIndex], samples[i]);
                    }
                    ++nextIndex;
                }
            }
            reorderByImage(start, nextIndex);
            reorderByImage(nextIndex, end);
            return nextIndex;
        }

        // Reorder samples in [start, ..., end-1] by image index to improve cache performance
        void reorderByImage(int start, int end) {
            int nextIndex = start;
            static auto sampleComp = [](const Sample & a, const Sample & b) {
                if (a.index == b.index) {
                    if (a.pix[1] == b.pix[1]) return a.pix[0] < b.pix[0];
                    return a.pix[1] < b.pix[1];
                }
                return a.index < b.index;
            };
            sort(samples.begin() + start, samples.begin() + end, sampleComp);
        }

        std::vector<RTree::RNode>& nodes;
        std::vector<RTree::Distribution>& leafData;
        const int numParts;
        DataLoader<DataSource> dataLoader;

        bool verbose;
        int numImages, numFeatures, maxProbeOffset, minSamples, numThreads, samplesPerFeature, threshesPerFeature;

        // const int SAMPLES_PER_FEATURE = 60;
        std::vector<int> chosenImages;
        SampleVec samples;
    };

    struct FileDataSource {
        FileDataSource(
                const std::string& depth_dir,
                const std::string& part_mask_dir)
       : depthDir(depth_dir), partMaskDir(part_mask_dir) {
           reload();
       }
        
        void reload() {
            _data_paths[DATA_DEPTH].clear();
            _data_paths[DATA_PART_MASK].clear();
            using boost::filesystem::directory_iterator;
            // List directories
            for (auto it = directory_iterator(depthDir); it != directory_iterator(); ++it) {
                _data_paths[DATA_DEPTH].push_back(it->path().string());
            }
            std::sort(_data_paths[DATA_DEPTH].begin(), _data_paths[DATA_DEPTH].end());
            for (auto it = directory_iterator(partMaskDir); it != directory_iterator(); ++it) {
                _data_paths[DATA_PART_MASK].push_back(it->path().string());
            }
            std::sort(_data_paths[DATA_PART_MASK].begin(), _data_paths[DATA_PART_MASK].end());
        }

        int size() const {
            return _data_paths[0].size();
        }

        const std::array<cv::Mat, 2>& load(int idx) {
            thread_local std::array<cv::Mat, 2> arr;
            thread_local int last_idx = -1;
            if (idx != last_idx) {
                arr[0] = cv::imread(_data_paths[idx][0], IMREAD_FLAGS[0]);
                arr[1] = cv::imread(_data_paths[idx][1], IMREAD_FLAGS[1]);
                last_idx = idx;
            }
            return arr;
        }

        void serialize(std::ostream& os) {
            os.write("SRC_FILE", 8);
            util::write_bin(os, depthDir.size());
            os.write(depthDir.c_str(), depthDir.size());
            util::write_bin(os, partMaskDir.size());
            os.write(depthDir.c_str(), depthDir.size());
        }

        void deserialize(std::istream& is) {
            char marker[8];
            is.read(marker, 8);
            if (strncmp(marker, "SRC_FILE", 8)) {
                std::cerr << "ERROR: Invalid file data source specified in stored samples file\n";
                return;
            }
            size_t sz;
            util::read_bin(is, sz);
            depthDir.resize(sz);
            is.read(&depthDir[0], sz);
            util::read_bin(is, sz);
            partMaskDir.resize(sz);
            is.read(&partMaskDir[1], sz);
            reload();
        }

        std::vector<std::string> _data_paths[2];
        std::string depthDir, partMaskDir;
    };

    struct AvatarDataSource {
        AvatarDataSource(AvatarModel& ava_model, 
                         AvatarPoseSequence& pose_seq,
                         CameraIntrin& intrin,
                         cv::Size& image_size,
                         int num_images,
                         const int* part_map)
            : avaModel(ava_model), poseSequence(pose_seq), intrin(intrin), imageSize(image_size),
              numImages(num_images), partMap(part_map) {
            seq.reserve(poseSequence.numFrames);
            for (int i = 0; i < poseSequence.numFrames; ++i) {
                seq.push_back(i);
            }
            for (int i = 0; i < poseSequence.numFrames; ++i) {
                int r = random_util::randint<size_t>(i, poseSequence.numFrames - 1);
                if (r != i) std::swap(seq[r], seq[i]);
            }
        }

        int size() const {
            return numImages;
        }

        const std::array<cv::Mat, 2>& load(int idx) {
            thread_local std::array<cv::Mat, 2> arr;
            thread_local int last_idx = -1;
            thread_local Avatar ava(avaModel);
            if (idx != last_idx) {
                last_idx = idx;
                if (poseSequence.numFrames) {
                    // random_util::randint<size_t>(0, poseSequence.numFrames - 1)
                    int seqid = seq[idx % poseSequence.numFrames];
                    poseSequence.poseAvatar(ava, seqid);
                    ava.r[0].setIdentity();
                    ava.randomize(false, true, true, idx);
                } else {
                    ava.randomize(true, true, true, idx);
                }
                ava.update();
                AvatarRenderer renderer(ava, intrin);
                arr[0] = renderer.renderDepth(imageSize),
                arr[1] = renderer.renderPartMask(imageSize, partMap);
            }
            return arr;
        }

        // Warning: serialization is incomplete, still need to load same avatar model, pose sequence, etc.
        void serialize(std::ostream& os) {
            os.write("SRC_AVATAR", 10);
            util::write_bin(os, seq.size());
            for (int i : seq) {
                util::write_bin(os, i);
            }
        }

        void deserialize(std::istream& is) {
            char marker[10];
            is.read(marker, 10);
            if (strncmp(marker, "SRC_AVATAR", 10)) {
                std::cerr << "ERROR: Invalid avatar data source specified in stored samples file\n";
                return;
            }

            seq.clear();
            size_t sz;
            util::read_bin(is, sz);
            seq.reserve(sz);
            int x;
            for (int i = 0; i < sz; ++i) {
                util::read_bin(is, x);
                seq.push_back(x);
            }
        }

        int numImages;
        cv::Size imageSize;
        AvatarModel& avaModel;
        AvatarPoseSequence& poseSequence;
        CameraIntrin& intrin;
        std::vector<int> seq;
        const int* partMap;
    };
 
    RTree::RTree(int num_parts) : numParts(num_parts) {}
    RTree::RTree(const std::string & path) {
        if (!loadFile(path)) {
            fprintf(stderr, "ERROR: RTree failed to initialize from %s\n", path.c_str());
        }
    }

    bool RTree::loadFile(const std::string & path) {
        size_t nNodes, nLeafs;
        std::ifstream ifs(path);
        if (!ifs) return false;

        ifs >> nNodes >> nLeafs >> numParts;
        nodes.resize(nNodes);
        leafData.resize(nLeafs);

        for (size_t i = 0; i < nNodes; ++i) {
            ifs >> nodes[i].leafid;
            if (nodes[i].leafid < 0) {
                ifs >> nodes[i].lnode >>
                       nodes[i].rnode >>
                       nodes[i].thresh >>
                       nodes[i].u[0] >>
                       nodes[i].u[1] >>
                       nodes[i].v[0] >>
                       nodes[i].v[1];
            }
        }

        for (size_t i = 0; i < nLeafs; ++i) {
            leafData[i].resize(numParts);
            for (int j = 0 ; j < numParts; ++j){
                ifs >> leafData[i](j);
            }
        }
        
        return true;
    }

    bool RTree::exportFile(const std::string & path) {
        std::ofstream ofs(path);
        ofs << std::fixed << std::setprecision(11);
        ofs << nodes.size() << " " << leafData.size() << " " << numParts << "\n";
        for (size_t i = 0; i < nodes.size(); ++i) {
            ofs << " " << nodes[i].leafid;
            if (nodes[i].leafid < 0) {
                ofs << "  " << nodes[i].lnode <<
                        " " << nodes[i].rnode <<
                        " " << nodes[i].thresh <<
                        " " << nodes[i].u[0] <<
                        " " << nodes[i].u[1] <<
                        " " << nodes[i].v[0] <<
                        " " << nodes[i].v[1];
            }
            ofs << "\n";
        }
        for (size_t i = 0; i < leafData.size(); ++i) {
            ofs << " ";
            for (int j = 0 ; j < numParts; ++j){
                ofs << leafData[i](j) << " ";
            }
            ofs << "\n";
        }
        ofs.close();
        return true;
    }

     RTree::Distribution RTree::predictRecursive(int nodeid, const cv::Mat& depth, const Vec2i& pix) {
         auto& node = nodes[nodeid];
         if (node.leafid == -1) {
             if (scoreByFeature(depth, pix, node.u, node.v) < node.thresh) {
                 return predictRecursive(node.lnode, depth, pix);
             } else {
                 return predictRecursive(node.rnode, depth, pix);
             }
         } else {
             return leafData[node.leafid];
         }
     }

    RTree::Distribution RTree::predict(const cv::Mat& depth, const Vec2i& pix) {
        return predictRecursive(0, depth, pix);
    }

    std::vector<cv::Mat> RTree::predict(const cv::Mat& depth) {
        std::vector<cv::Mat> result;
        result.reserve(numParts);
        for (int i = 0; i < numParts; ++i) {
            result.emplace_back(depth.size(), CV_32F);
            result[i].setTo(0.f);
        }
        Vec2i pix;
        Distribution distr;
        std::vector<float*> ptr(numParts);
        for (int r = 0; r < depth.rows; ++r) {
            pix(1) = r;
            for (int i = 0; i < numParts; ++i) {
                ptr[i] = result[i].ptr<float>(r);
            }
            const auto* inPtr = depth.ptr<float>(r);
            for (int c = 0; c < depth.cols; ++c) {
                if (inPtr[c] <= 0.f) continue; 
                pix(0) = c;
                distr = predictRecursive(0, depth, pix);
                for (int i = 0; i < numParts; ++i) {
                    ptr[i][c] = distr(i);
                }
            }
        }
        return result;
    }

    void RTree::train(const std::string& depth_dir,
                   const std::string& part_mask_dir,
                   int num_threads,
                   bool verbose,
                   int num_images,
                   int num_points_per_image,
                   int num_features,
                   int max_probe_offset,
                   int min_samples, 
                   int max_tree_depth,
                   int samples_per_feature,
                   int threshes_per_feature,
                   int max_images_loaded,
                   const std::string& samples_file,
                   bool generate_samples_file_only
               ) {
        nodes.reserve(1 << std::min(max_tree_depth, 22));
        FileDataSource dataSource(depth_dir, part_mask_dir);
        Trainer<FileDataSource> trainer(nodes, leafData, dataSource, numParts, max_images_loaded);
        bool shouldReadSamples = !samples_file.empty() && !generate_samples_file_only;
        if (shouldReadSamples) {
            trainer.readSamples(samples_file, verbose);
        }
        trainer.train(num_images, num_points_per_image, num_features,
                max_probe_offset, min_samples, max_tree_depth, samples_per_feature, threshes_per_feature,
                num_threads, verbose, shouldReadSamples, generate_samples_file_only);
        if (generate_samples_file_only) {
            trainer.writeSamples(samples_file);
        }
    }

    void RTree::trainFromAvatar(AvatarModel& avatar_model,
                   AvatarPoseSequence& pose_seq,
                   CameraIntrin& intrin,
                   cv::Size& image_size,
                   int num_threads,
                   bool verbose,
                   int num_images,
                   int num_points_per_image,
                   int num_features,
                   int max_probe_offset,
                   int min_samples, 
                   int max_tree_depth,
                   int samples_per_feature,
                   int threshes_per_feature,
                   const int* part_map,
                   int max_images_loaded,
                   const std::string& samples_file,
                   bool generate_samples_file_only
               ) {
        nodes.reserve(1 << std::min(max_tree_depth, 22));
        AvatarDataSource dataSource(avatar_model, pose_seq, intrin, image_size, num_images, part_map);
        Trainer<AvatarDataSource> trainer(nodes, leafData, dataSource, numParts, max_images_loaded);
        bool shouldReadSamples = !samples_file.empty() && !generate_samples_file_only;
        if (shouldReadSamples) {
            trainer.readSamples(samples_file, verbose);
        }
        trainer.train(num_images, num_points_per_image, num_features,
                max_probe_offset, min_samples, max_tree_depth, samples_per_feature,
                threshes_per_feature, num_threads, verbose, shouldReadSamples, generate_samples_file_only);
        if (generate_samples_file_only) {
            trainer.writeSamples(samples_file);
        }
    }
}
