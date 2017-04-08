#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <opencv2/opencv.hpp>

namespace sm {

const std::string save_path = "../output/";
typedef unsigned long long       bitmap;
typedef short                    cost_type;
typedef std::vector<cost_type>   buffer_line;
typedef std::vector<buffer_line> buffer;

struct Settings {
  int census_wing; // size of census rank kernel
  int census_size; //
  int max_d;       // maximal disparity value
  int width;       // image width
  int p1;          //
  int p2;          //
  Settings () {
    census_wing = 3;
    census_size = census_wing * 2 + 1;
    max_d = 128;
    width = -1;
    p1    = 3;
    p2    = 10;
  }
};

class StereoMatcher {

public:
  StereoMatcher () {}
  // Computes disparity map of two images.
  void Process(const cv::Mat& base, const cv::Mat& match);

private:
  // Computes disparity line by line
  void ProcessLine(const uchar* base, const uchar* match, uchar* res, int y);
  // Computes census rank transformation for one line.
  void InitializeCensusRank(int y);
  // Recomputes census rank for upcoming line.
  void RecomputeCensusRank();
  // Aggregates cost in given direction
  void AggregateCost(int x_from, int x_to, int delta_x,
                     int x_dir, int x_edge, int y_cur, int y_prev);

private:
  /// Resources:
  Settings settings_;
  // [census_size][width] arrays storing last census_size lines of image needed
  // to compute census rank transformation
  cv::Mat base_buffer_;
  cv::Mat match_buffer_;

  // Census rank for base image line
  std::vector<bitmap> base_census_;  
  // Census rank for match image line
  std::vector<bitmap> match_census_;
  // [max_d][width] array storing basic cost 
  buffer basic_cost_;
  // [max_d][width] array storing aggregated cost
  buffer aggregated_cost_;
  // [2][max_d][width] array storing aggregated cost for previous and current 
  // lines
  std::vector<buffer> aggregation_buffer_;
};

void StereoMatcher::Process(const cv::Mat& base, const cv::Mat& match) {
  /// 1. Initialization
  // reading input images

  // resulting image. will be filled line by line
  cv::Mat result(cv::Mat::zeros(base.rows, base.cols, base.type()));

  // initializing resources
  base_buffer_ = cv::Mat::zeros(settings_.census_size, base.cols, base.type());
  match_buffer_ = cv::Mat::zeros(settings_.census_size, base.cols, base.type());
  settings_.width = base.cols;

  const cost_type max_basic_cost = settings_.census_size * settings_.census_size;
  basic_cost_.assign(settings_.max_d, 
                     buffer_line(settings_.width, max_basic_cost));

  aggregated_cost_.assign(settings_.max_d, buffer_line(settings_.width, 0));
  aggregation_buffer_.assign(2,
      buffer(settings_.max_d, buffer_line(settings_.width, 0)));

  /// 2. Processing
  for (size_t y = 0; y < size_t(base.rows); ++y) {
    ProcessLine(base.ptr(y), match.ptr(y), result.ptr(y), y);
  }

  /// 3. Saving answer
  cv::normalize(result, result, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imwrite(save_path + "disparity_map.png", result);

  // will be impossible to apply median filter with online algorithm,
  // without little tricks
  cv::Mat result_blurred;
  cv::medianBlur(result, result_blurred, 3);
  cv::imwrite(save_path + "blurred_disparity_map.png", result_blurred);
}

cost_type ComputeHammingDistance(const bitmap& a, const bitmap& b) {
#ifdef _MSC_VER
  return cost_type(__popcnt64(a ^ b));
#else 
  return cost_type(__builtin_popcountll(a ^ b));
#endif
};

void StereoMatcher::AggregateCost(int x_from, int x_to, int delta_x,
                                  int x_dir, int x_edge, int y_cur, int y_prev) {

  static const cost_type inf = std::numeric_limits<cost_type>::max();
  buffer& cur = aggregation_buffer_[y_cur];
  buffer& prev = aggregation_buffer_[y_prev];

  for (int x = x_from; x != x_to; x += delta_x) {
    for (int d = 0; d < settings_.max_d; ++d) {
      cost_type cur_val = basic_cost_[d][x];

      if (x == x_edge) {
        cur[d][x] = cur_val;
        continue;
      }

      const cost_type m1 = cur[d][x + x_dir];
      const cost_type m2 = (d != settings_.max_d - 1 ? 
                            cur[d + 1][x + x_dir] + settings_.p1 : inf);
      const cost_type m3 = (d != 0 ?                   
                            cur[d - 1][x + x_dir] + settings_.p1 : inf);

      cost_type min_pred = inf;
      for (int temp_d = 0; temp_d < settings_.max_d; ++temp_d) {
        min_pred = std::min(min_pred, cur[temp_d][x + x_dir]);
      }
      const cost_type m4 = min_pred + settings_.p2;

      cur_val += std::min(m1,
                 std::min(m2,
                 std::min(m3,
                          m4)));

      cur_val -= min_pred;
      cur[d][x] = cur_val;
      prev[d][x] += cur_val;
      aggregated_cost_[d][x] += prev[d][x];
    }
  }
}

void StereoMatcher::ProcessLine(const uchar* base, const uchar* match,
                                uchar* res, int y) {

  int local_y = y % settings_.census_size;

  memcpy(base_buffer_.ptr(local_y), base, settings_.width);
  memcpy(match_buffer_.ptr(local_y), match, settings_.width);

  if (y < settings_.census_size - 1) {
    return;
  } else if (y == settings_.census_size - 1) {
    InitializeCensusRank(settings_.census_wing);
  } else {
    InitializeCensusRank((y + settings_.census_wing) % settings_.census_size);
    //RecomputeCensusRank();
  }

  // disparity cost computing
  const int x_from = settings_.census_wing;
  const int x_to   = settings_.width - settings_.census_wing;
  const int d_from = -settings_.max_d / 2;
  const int d_to   = settings_.max_d / 2;
  for (int x = x_from; x < x_to; ++x) {
    for (int d = d_from; d < d_to; ++d) {
      int x_match = x + d;
      if (x_match > x_from && x_match < x_to) {
        basic_cost_[d - d_from][x] =
            ComputeHammingDistance(base_census_[x], match_census_[x_match]);
      }
    }
  }

  // cost aggregation

  if (y == settings_.census_size - 1) {
    const int y_0 = y % 2;
    buffer& cur = aggregation_buffer_[y_0];
    for (int x = x_from; x < x_to; ++x) {
      for (int d = 0; d < settings_.max_d; ++d) {
        cur[d][x] = basic_cost_[d][x];
      }
    }
  } else {
    const int y_0 = y % 2;
    const int y_1 = 1 - y_0;

    for (int i = 0; i < aggregated_cost_.size(); ++i) {
      std::fill(begin(aggregated_cost_[i]), end(aggregated_cost_[i]), 0);
    }
    // left to right
    AggregateCost(x_from, x_to, 1, 
                  -1, x_from, y_0, y_1);

    // right to left
    AggregateCost(x_to - 1, x_from - 1, -1,
                  1, x_to - 1, y_0, y_1);
                  
    // to left top 
    AggregateCost(x_from, x_to, 1, 
                  -1, x_from, y_0, y_0);
  }
  // saving answer
  for (int x = x_from; x < x_to; ++x) {
    cost_type min_val = std::numeric_limits<cost_type>::max();
    for (int d = 0; d < settings_.max_d; ++d) {
      if (aggregated_cost_[d][x] < min_val) {
        res[x] = d;
        min_val = cost_type(aggregated_cost_[d][x]);
      }
    }
  }
}

void StereoMatcher::InitializeCensusRank(int y) {
  base_census_.resize(settings_.width);
  match_census_.resize(settings_.width);

  int& wdt = settings_.width;
  int& cwg = settings_.census_wing;
  int& sz  = settings_.census_size;

  for (int x = cwg; x < wdt - cwg; ++x) {
    bitmap base_bitmap = 0;
    bitmap match_bitmap = 0;
    int id = 0;
    const uchar base_val = base_buffer_.at<uchar>(y, x);
    const uchar match_val = match_buffer_.at<uchar>(y, x);

    for (int i = 0; i < sz; ++i) {
      const uchar* p_base = base_buffer_.ptr(i);
      const uchar* p_match = match_buffer_.ptr(i);
      for (int j = x - cwg; j < x + cwg + 1; ++j) {
        if (p_base[j] < base_val) {
          base_bitmap |= (1ll << id);
        }
        if (p_match[j] < match_val) {
          match_bitmap |= (1ll << id);
        }
        id++;
      }
    }
    base_census_[x] = base_bitmap;
    match_census_[x] = match_bitmap;
  }
}

void StereoMatcher::RecomputeCensusRank() {
  // recomputes census rank with O(census_wing) instead of O(census_wing^2)
}

} // namespace sm

int main(int argc, char* argv[]) {

  std::string path_base;
  std::string path_match;

  if (argc < 2) {
    path_base = "../testdata/cones1.png";
    path_match = "../testdata/cones2.png";
  } else {
    path_base = argv[1];
    path_match = argv[2];
  }
  cv::Mat base = cv::imread(path_base, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat match = cv::imread(path_match, CV_LOAD_IMAGE_GRAYSCALE);

  if (base.data && match.data) {
    sm::StereoMatcher solver;
    solver.Process(base, match);
  } else {
    std::cout << "Failed to read images" << std::endl;
  }
  return 0;
}
