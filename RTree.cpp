#include "RTree.h"

#include <fstream>
#include <chrono>
#include <cstdio>
#include <random>
#include <deque>
#include <mutex>
#include <iomanip>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "boost/filesystem.hpp"

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
    inline float getDepth(const cv::Mat& depth_image, const Eigen::Vector2i& point) {
        static const float BACKGROUND_DEPTH = 20.f;
        if (point.y() < 0 || point.x() < 0 ||
                point.y() >= depth_image.rows || point.x() >= depth_image.cols)
            return BACKGROUND_DEPTH;
        float depth = depth_image.at<float>(point.y(), point.x());
        if (depth <= 0.0) return BACKGROUND_DEPTH;
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
            uti << static_cast<int>(std::round(ut.x())), static_cast<int>(std::round(ut.y()));
            vti << static_cast<int>(std::round(vt.x())), static_cast<int>(std::round(vt.y()));
            uti += pix; vti += pix;
            
            return (getDepth(depth_image, uti) - getDepth(depth_image, vti));
        }
}

namespace ark {
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
    }; 

    template<class DataSource>
    struct DataLoader {
        DataSource& dataSource;
        DataLoader(DataSource& data_source, int max_images_loaded) : dataSource(data_source), maxImagesLoaded(max_images_loaded) {}

        bool preload(const std::vector<Sample>& samples, int a, int b, int numThreads) {
            std::vector<int> images;
            for (const auto& sample : samples) {
                images.push_back(sample.index);
            }
            std::sort(images.begin(), images.end());
            images.resize(std::unique(images.begin(), images.end()) - images.begin());
            if (images.size() > maxImagesLoaded) {
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

        std::array<cv::Mat, 2> get(const Sample& sample) const {
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
                   int num_threads, bool verbose) {
            this->verbose = verbose;

            // Initialize
            initTraining(num_images, num_points_per_image, max_tree_depth);
            
            if (verbose) {
                std::cerr << "Init RTree training with maximum depth " << max_tree_depth << "\n";
            }

            // Train
            numFeatures = num_features;
            maxProbeOffset = max_probe_offset;
            minSamples = min_samples;
            numThreads = num_threads;
            nodes.resize(1);
            trainFromNode(nodes[0], 0, static_cast<int>(samples.size()), max_tree_depth);
            if (verbose) {
                std::cerr << "Training finished\n";
            }
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

            dataLoader.preload(samples, start, end, numThreads);
            int featureCount = numFeatures;
            Eigen::VectorXf bestInfoGains(numThreads, 1);
            Eigen::VectorXf bestThreshs(numThreads, 1);
            bestInfoGains.setConstant(-FLT_MAX);
            std::vector<Feature> bestFeatures(numThreads);

            // Mapreduce-ish
            auto worker = [&](int thread_id) {
                float& bestInfoGain = bestInfoGains(thread_id);
                float& bestThresh = bestThreshs(thread_id);
                Feature& bestFeature = bestFeatures[thread_id];
                Feature feature;
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(trainMutex);
                        if (featureCount <= 0) break;
                        --featureCount;
                    }

                    // Create random feature in-place
                    feature.u.x() = random_util::uniform(1.0, maxProbeOffset) * (random_util::randint(0, 2) * 2 - 1);
                    feature.u.y() = random_util::uniform(1.0, maxProbeOffset) * (random_util::randint(0, 2) * 2 - 1);
                    feature.v.x() = random_util::uniform(1.0, maxProbeOffset) * (random_util::randint(0, 2) * 2 - 1);
                    feature.v.y() = random_util::uniform(1.0, maxProbeOffset) * (random_util::randint(0, 2) * 2 - 1);

                    float optimalThresh;
                    float infoGain = optimalInformationGain(start, end, feature, &optimalThresh);

                    if (infoGain >= bestInfoGain) {
                        bestInfoGain = infoGain;
                        bestThresh = optimalThresh;
                        bestFeature = feature;
                    }
                }
            };

            std::vector<std::thread> threadMgr;
            for (int i = 0; i < numThreads; ++i) {
                threadMgr.emplace_back(worker, i);
            }
            
            int bestThreadId = 0;
            for (int i = 0; i < numThreads; ++i) {
                threadMgr[i].join();
                if (i && bestInfoGains(i) > bestInfoGains(bestThreadId)) {
                    bestThreadId = i;
                }
            }
            
            int mid = split(start, end, bestFeatures[bestThreadId], bestThreshs(bestThreadId));

            if (verbose) {
                std::cerr << "> Best info gain " << bestInfoGains(bestThreadId) << ", thresh " << bestThreshs(bestThreadId) << ", feature.u " << bestFeatures[bestThreadId].u.x() << "," << bestFeatures[bestThreadId].u.y() <<", features.v" << bestFeatures[bestThreadId].u.x() << "," << bestFeatures[bestThreadId].u.y() << "\n";
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
            node.thresh = bestThreshs(bestThreadId);
            node.u = bestFeatures[bestThreadId].u;
            node.v = bestFeatures[bestThreadId].v;

            node.lnode = static_cast<int>(nodes.size());
            nodes.emplace_back();
            trainFromNode(nodes.back(), start, mid, depth - 1);

            node.rnode = static_cast<int>(nodes.size());
            nodes.emplace_back();
            trainFromNode(nodes.back(), mid, end, depth - 1);
        }

        // Compute information gain (mutual information scaled and shifted) by choosing optimal threshold
        float optimalInformationGain(int start, int end, const Feature& feature, float* optimal_thresh) {

            // Initially everything is in left set
            RTree::Distribution distLeft(numParts), distRight(numParts);
            distLeft.setZero();
            distRight.setZero();

            // Compute scores
            std::vector<std::pair<float, int> > samplesByScore;
            for (int i = start; i < end; ++i) {
                const Sample& sample = samples[i];
                auto dataArr = dataLoader.get(sample);
                samplesByScore.emplace_back(
                        scoreByFeature(dataArr[DATA_DEPTH],
                            sample.pix, feature.u, feature.v) , i);
                uint8_t samplePart = dataArr[DATA_PART_MASK]
                    .template at<uint8_t>(sample.pix[1], sample.pix[0]);
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
            float bestInfoGain = -FLT_MAX;
            *optimal_thresh = samplesByScore[0].first;
            size_t step = std::max<size_t>(samplesByScore.size() / SAMPLES_PER_FEATURE, 1);
            float lastScore = -FLT_MAX;
            for (size_t i = 0; i < samplesByScore.size()-1; ++i) {
                // Update distributions for left, right sets
                const Sample& sample = samples[samplesByScore[i].second];
                auto samplePart = dataLoader.get(sample)[DATA_PART_MASK]
                    .template at<uint8_t>(sample.pix.y(), sample.pix.x());
                distLeft[samplePart] -= 1.f;
                distRight[samplePart] += 1.f;
                if (i%step != 0 || lastScore == samplesByScore[i].first)
                    continue;
                lastScore = samplesByScore[i].first;

                float left_entropy = entropy(distLeft / distLeft.sum());
                float right_entropy = entropy(distRight / distRight.sum());
                // Compute the information gain
                float infoGain = - ((end - start - i - 1) * left_entropy
                                 + (i+1)                  * right_entropy);
                if (infoGain > 0) {
                    std::cerr << "FATAL: Possibly overflow detected during training, exiting. Internal data: left entropy "
                        << left_entropy << " right entropy "
                        << right_entropy << " information gain "
                        << infoGain<< "\n";
                    std::exit(2);
                }
                if (infoGain > bestInfoGain) {
                    // If better then update optimal thresh to between samples
                    *optimal_thresh = random_util::uniform(samplesByScore[i].first, samplesByScore[i+1].first);
                    bestInfoGain = infoGain;
                }
            }
            return bestInfoGain;
        }

        void initTraining(int num_images, int num_points_per_image, int max_tree_depth) {
            // 1. Choose num_images random images u.a.r. from given image list
            std::vector<int> allImages(dataLoader.dataSource.size());
            std::iota(allImages.begin(), allImages.end(), 0);
            chosenImages = allImages.size() > static_cast<size_t>(num_images) ?
                random_util::choose(allImages, num_images) : std::move(allImages);

            // 2. Choose num_points_per_image random foreground pixels from each image,
            for (size_t i = 0; i < chosenImages.size(); ++i) {
                if (verbose && i % 10 == 9) {
                    std::cerr << "Preprocessing data: " << i+1 << " of " << num_images << "\n";
                }
                cv::Mat mask = dataLoader.get(Sample(chosenImages[i], 0, 0))[DATA_PART_MASK];
                // cv::Mat mask2 = dataLoader.get(Sample(chosenImages[i], 0, 0))[DATA_PART_MASK];
                // cv::hconcat(mask, mask2, mask);
                // cv::resize(mask, mask, mask.size() / 2);
                // cv::imshow("MASKCat", mask);
                // cv::waitKey(0);
                std::vector<RTree::Vec2i> candidates;
                for (int r = 0; r < mask.rows; ++r) {
                    auto* ptr = mask.ptr<uint8_t>(r);
                    for (int c = 0; c < mask.cols; ++c) {
                        if (ptr[c] != 255) {
                            candidates.emplace_back();
                            candidates.back() << c, r;
                        }
                    }
                }
                std::vector<RTree::Vec2i> chosenCandidates =
                    (candidates.size() > static_cast<size_t>(num_points_per_image)) ?
                    random_util::choose(candidates, num_points_per_image) : std::move(candidates);
                for (auto& v : chosenCandidates) {
                    samples.emplace_back(chosenImages[i], v);
                }
            }

            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(samples.begin(), samples.end(), g);
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
            return nextIndex;
        }

        std::vector<RTree::RNode>& nodes;
        std::vector<RTree::Distribution>& leafData;
        const int numParts;
        DataLoader<DataSource> dataLoader;

        bool verbose;
        int numFeatures, maxProbeOffset, minSamples, numThreads;

        // TODO: Make this a configurable argument
        const int SAMPLES_PER_FEATURE = 60;
        std::vector<int> chosenImages;
        std::vector<Sample> samples;
    };

    struct FileDataSource {
        FileDataSource(
                const std::string& depth_dir,
                const std::string& part_mask_dir) {
            using boost::filesystem::directory_iterator;
            // List directories
            for (auto it = directory_iterator(depth_dir); it != directory_iterator(); ++it) {
                _data_paths[DATA_DEPTH].push_back(it->path().string());
            }
            std::sort(_data_paths[DATA_DEPTH].begin(), _data_paths[DATA_DEPTH].end());
            for (auto it = directory_iterator(part_mask_dir); it != directory_iterator(); ++it) {
                _data_paths[DATA_PART_MASK].push_back(it->path().string());
            }
            std::sort(_data_paths[DATA_PART_MASK].begin(), _data_paths[DATA_PART_MASK].end());
        }

        int size() const {
            return _data_paths[0].size();
        }

        std::array<cv::Mat, 2> load(int idx) {
            return {
                    cv::imread(_data_paths[idx][0], IMREAD_FLAGS[0]),
                    cv::imread(_data_paths[idx][1], IMREAD_FLAGS[1])
            };
        }
        std::vector<std::string> _data_paths[2];
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

        std::array<cv::Mat, 2> load(int idx) {
            thread_local Avatar ava(avaModel);
            ava.randomize(true, true, true, idx);

            if (poseSequence.numFrames) {
                // random_util::randint<size_t>(0, poseSequence.numFrames - 1)
                poseSequence.poseAvatar(ava, seq[idx % poseSequence.numFrames]);
                ava.r[0].setIdentity();
                ava.randomize(false, true, true, idx);
            } else {
                ava.randomize(true, true, true, idx);
            }
            ava.update();
            AvatarRenderer renderer(ava, intrin);
            return {
                renderer.renderDepth(imageSize),
                renderer.renderPartMask(imageSize, partMap)
            };
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
                   int max_images_loaded) {
        nodes.reserve(1 << std::min(max_tree_depth, 22));
        FileDataSource dataSource(depth_dir, part_mask_dir);
        Trainer<FileDataSource> trainer(nodes, leafData, dataSource, numParts, max_images_loaded);
        trainer.train(num_images, num_points_per_image, num_features,
                max_probe_offset, min_samples, max_tree_depth, num_threads, verbose);
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
                   const int* part_map,
                   int max_images_loaded) {
        nodes.reserve(1 << std::min(max_tree_depth, 22));
        AvatarDataSource dataSource(avatar_model, pose_seq, intrin, image_size, num_images, part_map);
        Trainer<AvatarDataSource> trainer(nodes, leafData, dataSource, numParts, max_images_loaded);
        trainer.train(num_images, num_points_per_image, num_features,
                max_probe_offset, min_samples, max_tree_depth, num_threads, verbose);
    }
}
