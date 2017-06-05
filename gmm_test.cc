/*
 * gmm_test.cc
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "gmm.h"
#include "hog_grabcut.h"

using namespace cv;
using namespace gmm;

class GMM_Test : public ::testing::Test {
public:
  void SetUp() {
    img = imread("./test.png");
  }
  void TearDown() {}

public:
  Mat img;
};

TEST_F(GMM_Test, BasicBuild) {
  Mat mat;
  GMM<> gmm(mat);
}

TEST_F(GMM_Test, BUILDMASKBYMULTIRECTVEC) {
  std::vector<cv::Rect> rect(4);
  rect[0].x = 1407;
  rect[0].y = 478;
  rect[0].width = 98;
  rect[0].height = 196;
  rect[1].x = 1133;
  rect[1].y = 409;
  rect[1].width = 87;
  rect[1].height = 175;
  rect[2].x = 1212;
  rect[2].y = 416;
  rect[2].width = 77;
  rect[2].height = 153;
  rect[3].x = 873;
  rect[3].y = 95;
  rect[3].width = 211;
  rect[3].height = 422;

  Mat mask;
  BuildMaskByRectVec(rect, img.size(), mask);
  HogGrabCut::ShowImage(img);
  Mat output;
  img.copyTo(output, mask & 1);
  HogGrabCut::ShowImage(output);
}
