#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

const string save_path = "../output/";
typedef unsigned long long bitmap;

uchar compute_hamming_distance(const bitmap& a, const bitmap& b) {
  uchar ret = 0;
  bitmap c = a ^ b;
  for (size_t i = 0; i < 48; ++i) {
    ret += bool((1LL << i) & c);
  }
  return ret;
}

vector<uchar> compute_census(const Mat& img, const int& y,
                             const int& x, const int& k) {
  vector<uchar> ret(48);

  uchar val = img.ptr(y)[x];
  int id = 0;

  for (int i = y - k; i < y + k + 1; ++i) {
    const uchar* p = img.ptr(i);
    for (int j = x - k; j < x + k + 1; ++j) {
      if (i == y && j == x)
        continue;
      if (p[j] < val)
        ret[id] = 1;
      id++;
    }
  }
  return ret;
}

bitmap compute_census_bitmap(const Mat& img, const int& y,
                             const int& x, const int& k) {
  bitmap ret = 0;

  uchar val = img.ptr(y)[x];
  int id = 0;

  for (int i = y - k; i < y + k + 1; ++i) {
    const uchar* p = img.ptr(i);
    for (int j = x - k; j < x + k + 1; ++j) {
      if (i == y && j == x)
        continue;
      if (p[j] < val)
        ret |= (1ll << id);
      id++;
    }
  }
  return ret;
}

void log_disp_map(const vector<vector<vector<int> > >& cost, string nm) {

  size_t max_d = cost.size();
  size_t h = cost[0].size();
  size_t w = cost[0][0].size();

  Mat ret(Mat::zeros(h, w, CV_32SC1));

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      int d_cost = cost[0][y][x];
      int d_val  = 0;
      for (size_t d = 1; d < max_d; ++d) {
        if (cost[d][y][x] < d_cost) {
          d_cost = cost[d][y][x];
          d_val = d;
        }
      }
      ret.at<int>(y, x) = d_val;
    }
  }

  normalize(ret, ret, 0, 255, NORM_MINMAX, CV_32SC1);
  Mat ret_normalized(Mat::zeros(h, w, CV_8UC1));

  ret.convertTo(ret_normalized, CV_8UC1);

  imwrite(save_path + "normalized_disp_map_" + nm + ".png", ret_normalized);

  medianBlur(ret_normalized, ret_normalized, 3);
  imwrite(save_path + "blurred_disp_map_" + nm + ".png", ret_normalized);
}

vector<vector<vector<int> > > compute_census_rand_basic_cost(
    const Mat& b, const Mat& m, int max_d, int k = 3) {
  int w = b.cols;
  int h = b.rows;

  //vector<vector<vector<uchar> > > fb(h, vector<vector<uchar> >(w));
  //vector<vector<vector<uchar> > > fm(h, vector<vector<uchar> >(w));
  vector<vector<bitmap> > fb(h, vector<bitmap>(w));
  vector<vector<bitmap> > fm(h, vector<bitmap >(w));

  vector<vector<vector<int> > > basic_cost(max_d,
      vector<vector<int> >(h, vector<int>(w, 100000)));

  // k - window's wing size
  for (int y = k; y < h - k; ++y) {
    for (int x = k; x < w - k; ++x) {
      //fb[y][x] = compute_census(b, y, x, k);
      //fm[y][x] = compute_census(m, y, x, k);
      fb[y][x] = compute_census_bitmap(b, y, x, k);
      fm[y][x] = compute_census_bitmap(m, y, x, k);

    }
  }

  Mat disp_map(Mat::zeros(h, w, CV_8UC1));

  for (int y = k; y < h - k; ++y) {
    for (int x = k ; x < w - k; ++x) {
      int value = 0;
      uchar min_dist = 100;
      for (int d = - max_d / 2; d < max_d / 2; ++d) {
        int x_d = x + d;
        if (x_d < k + 1 || x_d > w - k - 1)
          continue;
        uchar cur_dist = compute_hamming_distance(fb[y][x], fm[y][x_d]);
        if (cur_dist < min_dist) {
          min_dist = cur_dist;
          value = d;
        }
        basic_cost[d + max_d / 2][y][x] = cur_dist;
      }
      disp_map.at<uchar>(y, x) = value + max_d / 2;
    }
  }
  return basic_cost;
}


void aggregate_cost(vector<vector<vector<int> > >& sum,
                    vector<vector<vector<int> > >& a) {
  int max_val = 0;
  for (size_t i = 0; i < sum.size(); ++i) {
    for (size_t j = 0; j < sum[0].size(); ++j) {
      for (size_t k = 0; k < sum[0][0].size(); ++k) {
        sum[i][j][k] += a[i][j][k];
        max_val = std::max(a[i][j][k], max_val);
        a[i][j][k] = 0;
      }
    }
  }
}


void compute_cost_aggregation(const vector<vector<vector<int> > >& basic_cost,
                              int p1, int p2) {

  log_disp_map(basic_cost, "0_basic_cost");

  size_t max_d = basic_cost.size();
  size_t h = basic_cost[0].size();
  size_t w = basic_cost[0][0].size();

  vector<vector<vector<int> > > cur_cost(max_d,
      vector<vector<int> >(h, vector<int>(w, 0)));
  vector<vector<vector<int> > > sum_cost(max_d,
      vector<vector<int> >(h, vector<int>(w, 0)));

  const int inf = numeric_limits<int>::max();

  // left to right
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      for (size_t d = 0; d < max_d; ++d) {
        int cur_val = basic_cost[d][y][x];

        if (x == 0) {
          cur_cost[d][y][x] = cur_val;
          continue;
        }

        int m1 = cur_cost[d][y][x - 1];
        int m2 = (d != max_d - 1 ? cur_cost[d + 1][y][x - 1] + p1 : inf);
        int m3 = (d != 0         ? cur_cost[d - 1][y][x - 1] + p1 : inf);

        int min_pred = numeric_limits<int>::max();
        for (size_t temp_d = 0; temp_d < max_d; ++temp_d) {
          min_pred = std::min(min_pred, cur_cost[temp_d][y][x - 1]);
        }
        int m4 = min_pred + p2;

        cur_val += std::min(m1,
                   std::min(m2,
                   std::min(m3,
                            m4)));
        cur_val -= min_pred;
        cur_cost[d][y][x] = cur_val;
      }
    }
  }
  log_disp_map(cur_cost, "self_1");
  aggregate_cost(sum_cost, cur_cost);
  log_disp_map(sum_cost, "aggregated_1");

  // right to left
  for (size_t y = 0; y < h; ++y) {
    for (int x = w - 1; x >= 0; --x) {
      for (size_t d = 0; d < max_d; ++d) {
        int cur_val = basic_cost[d][y][x];

        if (x == int(w - 1)) {
          cur_cost[d][y][x] = cur_val;
          continue;
        }

        int m1 = cur_cost[d][y][x + 1];
        int m2 = (d != max_d - 1 ? cur_cost[d + 1][y][x + 1] + p1 : inf);
        int m3 = (d != 0         ? cur_cost[d - 1][y][x + 1] + p1 : inf);

        int min_pred = numeric_limits<int>::max();
        for (size_t temp_d = 0; temp_d < max_d; ++temp_d) {
          min_pred = std::min(min_pred, cur_cost[temp_d][y][x + 1]);
        }
        int m4 = min_pred + p2;

        cur_val += std::min(m1,
                   std::min(m2,
                   std::min(m3,
                            m4)));
        cur_val -= min_pred;
        cur_cost[d][y][x] = cur_val;
      }
    }
  }
  log_disp_map(cur_cost, "self_2");
  aggregate_cost(sum_cost, cur_cost);
  log_disp_map(sum_cost, "aggregated_2");

  // to top left
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      for (size_t d = 0; d < max_d; ++d) {
        int cur_val = basic_cost[d][y][x];

        if (x == 0 || y == 0) {
          cur_cost[d][y][x] = cur_val;
          continue;
        }

        int m1 = cur_cost[d][y - 1][x - 1];
        int m2 = (d != max_d - 1 ? cur_cost[d + 1][y - 1][x - 1] + p1 : inf);
        int m3 = (d != 0         ? cur_cost[d - 1][y - 1][x - 1] + p1 : inf);

        int min_pred = numeric_limits<int>::max();
        for (size_t temp_d = 0; temp_d < max_d; ++temp_d) {
          min_pred = std::min(min_pred, cur_cost[temp_d][y - 1][x - 1]);
        }
        int m4 = min_pred + p2;

        cur_val += std::min(m1,
                   std::min(m2,
                   std::min(m3,
                            m4)));
        cur_val -= min_pred;
        cur_cost[d][y][x] = cur_val;
      }
    }
  }
  log_disp_map(cur_cost, "self_3");
  aggregate_cost(sum_cost, cur_cost);
  log_disp_map(sum_cost, "aggregated_3");

  // to top right
  for (size_t y = 0; y < h; ++y) {
    for (int x = w - 1; x >= 0; --x) {
      for (size_t d = 0; d < max_d; ++d) {
        int cur_val = basic_cost[d][y][x];

        if (x == int(w - 1) || y == 0) {
          cur_cost[d][y][x] = cur_val;
          continue;
        }

        int m1 = cur_cost[d][y - 1][x + 1];
        int m2 = (d != max_d - 1 ? cur_cost[d + 1][y - 1][x + 1] + p1 : inf);
        int m3 = (d != 0         ? cur_cost[d - 1][y - 1][x + 1] + p1 : inf);

        int min_pred = numeric_limits<int>::max();
        for (size_t temp_d = 0; temp_d < max_d; ++temp_d) {
          min_pred = std::min(min_pred, cur_cost[temp_d][y - 1][x + 1]);
        }
        int m4 = min_pred + p2;

        cur_val += std::min(m1,
                            std::min(m2,
                                     std::min(m3,
                                              m4)));
        cur_val -= min_pred;
        cur_cost[d][y][x] = cur_val;
      }
    }
  }
  log_disp_map(cur_cost, "self_4");
  aggregate_cost(sum_cost, cur_cost);
  log_disp_map(sum_cost, "aggregated_4");

  // to top
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      for (size_t d = 0; d < max_d; ++d) {
        int cur_val = basic_cost[d][y][x];

        if (y == 0) {
          cur_cost[d][y][x] = cur_val;
          continue;
        }

        int m1 = cur_cost[d][y - 1][x];
        int m2 = (d != max_d - 1 ? cur_cost[d + 1][y - 1][x] + p1 : inf);
        int m3 = (d != 0         ? cur_cost[d - 1][y - 1][x] + p1 : inf);

        int min_pred = numeric_limits<int>::max();
        for (size_t temp_d = 0; temp_d < max_d; ++temp_d) {
          min_pred = std::min(min_pred, cur_cost[temp_d][y - 1][x]);
        }
        int m4 = min_pred + p2;

        cur_val += std::min(m1,
                            std::min(m2,
                                     std::min(m3,
                                              m4)));
        cur_val -= min_pred;
        cur_cost[d][y][x] = cur_val;
      }
    }
  }
  log_disp_map(cur_cost, "self_5");
  aggregate_cost(sum_cost, cur_cost);
  log_disp_map(sum_cost, "aggregated_5");
}

int main() {

  Mat base = imread("../testdata/cones1.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat match = imread("../testdata/cones2.png", CV_LOAD_IMAGE_GRAYSCALE);
  int max_d = 128;

  if (base.data && match.data) {
    vector<vector<vector<int> > > basic_cost = 
        compute_census_rand_basic_cost(base, match, max_d);
    compute_cost_aggregation(basic_cost, 3, 10);
  } else {
    std::cout << "Failed to read images" << std::endl;
  }
  return 0;
}
