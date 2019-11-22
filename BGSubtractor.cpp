#include "BGSubtractor.h"

#include <atomic>
#include <thread>
#include <opencv2/imgcodecs.hpp>
#include "Util.h"

namespace ark {
    namespace {
        cv::Mat ffill(
                const cv::Mat& background, const cv::Mat& image, int size, int num_threads,
                float nn_dist_thresh, float neighb_thresh,
                std::vector<std::array<int, 2> >* comps_by_size,
                cv::Point& top_left, cv::Point& bot_right) {
            cv::Mat absimg(image.size(), CV_8UC1);
            const uint8_t UNVISITED = 254, INVALID = 255;
            absimg.setTo(UNVISITED);
            std::vector<int> stk, curCompVis;
            int min_pts = std::max( background.rows * background.cols / 1000, 100);
            stk.reserve(background.rows * background.cols);
            curCompVis.reserve(background.rows * background.cols);
            int hi_bit = (1<<16);
            int lo_mask = hi_bit - 1;
            uint8_t compid = 0;
            if (comps_by_size != nullptr) {
                comps_by_size->reserve(254);
                comps_by_size->clear();
            }

            {
                std::atomic<int> row(0);
                auto check = [size, nn_dist_thresh, &background](int rr, int cc, const cv::Vec3f& val, uint8_t& invalid_bit) {
                    int minc = std::max(cc - size, 0), maxc = std::min(cc + size, background.cols - 1);
                    int minr = std::max(rr - size, 0), maxr = std::min(rr + size, background.rows - 1);
                    // float best_norm = std::numeric_limits<float>::max();
                    for (int r = minr; r <= maxr; ++r) {
                        const cv::Vec3f* rptr = background.ptr<cv::Vec3f>(r);
                        for (int c = minc; c <= maxc; ++c) {
                            const auto& neighb = rptr[c];
                            if (neighb[2] == 0.0) continue;
                            cv::Vec3f diff = neighb - val;
                            float norm = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                            if (norm < nn_dist_thresh) {
                                invalid_bit = INVALID;
                                return;
                            }
                        }
                    }
                };
                auto worker = [&] () {
                    int rr = 0;
                    while (true) {
                        rr = row++;
                        if (rr >= image.rows) break;
                        auto* ptr = image.ptr<cv::Vec3f>(rr);
                        auto* absptr = absimg.ptr<uint8_t>(rr);

                        for (int cc = 0 ; cc < image.cols; ++cc) {
                            if (ptr[cc][2] == 0.0) {
                                absptr[cc] = INVALID;
                                continue;
                            }
                            check(rr, cc, ptr[cc], absptr[cc]);
                        }
                    }
                };
                std::vector<std::thread> thds;
                for (int i = 0; i < num_threads; ++i) {
                    thds.emplace_back(worker);
                }
                for (int i = 0; i < num_threads; ++i) {
                    thds[i].join();
                }
            }

            auto maybe_visit = [&](int curr_r, int curr_c, const cv::Vec3f& curr_val, int new_r, int new_c, int new_id) {
                if (absimg.at<uint8_t>(new_r, new_c) == UNVISITED) {
                    const cv::Vec3f& val = image.at<cv::Vec3f>(new_r, new_c);
                    cv::Vec3f diff = curr_val - val;
                    if (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] > neighb_thresh) {
                        return;
                    }
                    absimg.at<uint8_t>(new_r, new_c) = compid;
                    curCompVis.push_back(new_id);
                    stk.push_back(curCompVis.back());
                }
            };

            for (int rr = 0; rr < image.rows; ++rr) {
                const cv::Vec3f* ptr = image.ptr<cv::Vec3f>(rr);
                auto* outptr = absimg.ptr<uint8_t>(rr);
                for (int cc = 0 ; cc < image.cols; ++cc) {
                    if (outptr[cc] != UNVISITED) continue;
                    outptr[cc] = compid;
                    stk.push_back((rr << 16) + cc);
                    curCompVis.clear();
                    curCompVis.push_back(stk.back());
                    while (stk.size()) {
                        int id = stk.back();
                        const int cur_r = (id >> 16), cur_c = (id & lo_mask);
                        stk.pop_back();
                        cv::Vec3f val = image.at<cv::Vec3f>(cur_r, cur_c);
                        if (cur_r > 0) maybe_visit(cur_r, cur_c, val, cur_r - 1, cur_c, id - hi_bit);
                        if (cur_r < background.rows - 1) maybe_visit(cur_r, cur_c, val, cur_r + 1, cur_c, id + hi_bit);
                        if (cur_c > 0) maybe_visit(cur_r, cur_c, val, cur_r, cur_c - 1, id - 1);
                        if (cur_c < background.cols - 1) maybe_visit(cur_r, cur_c, val, cur_r, cur_c + 1, id + 1);
                    }
                    if (curCompVis.size() < min_pts) {
                        for (int val : curCompVis) {
                            absimg.at<uint8_t>((val >> 16), (val & lo_mask)) = INVALID;
                        }
                    } else {
                        if (comps_by_size != nullptr) {
                            comps_by_size->push_back({static_cast<int>(curCompVis.size()), compid});
                        }
                        ++compid;
                    }
                    if (compid == UNVISITED) return absimg;
                }
            }

            top_left.x = image.cols - 1;
            top_left.y = image.rows - 1;
            bot_right.x = bot_right.y = 0;
            for (int rr = 0; rr < image.rows; ++rr) {
                auto* outptr = absimg.ptr<uint8_t>(rr);
                bool nz = false;
                for (int cc = 0 ; cc < image.cols; ++cc) {
                    if (outptr[cc] != INVALID) {
                        nz = true;
                        top_left.x = std::min(top_left.x, cc);
                        break;
                    }
                }
                if (nz) {
                    for (int cc = image.cols - 1; cc >= 0; --cc) {
                        if (outptr[cc] != INVALID) {
                            bot_right.x = std::max(bot_right.x, cc);
                            break;
                        }
                    }
                    if (top_left.y == image.rows - 1) top_left.y = rr;
                    bot_right.y = rr;
                }
            }
            if (comps_by_size != nullptr) {
                std::sort(comps_by_size->begin(), comps_by_size->end(), std::greater<std::array<int, 2> >());
            }
            return absimg;
        }
    }

    cv::Mat BGSubtractor::run(const cv::Mat& image, std::vector<std::array<int, 2> >* comps_by_size) {
        return ffill(background, image, 1, numThreads, 1200000.0 / (background.rows * background.cols) * nnDistThreshRel,
                                                        1200000.0 / (background.rows * background.cols) * neighbThreshRel,
                                    comps_by_size, topLeft, botRight);
    }
}
