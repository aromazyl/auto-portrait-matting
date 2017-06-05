/*
 * hog_grabcut_test.cc
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "hog_grabcut.h"

#include <gtest/gtest.h>

using namespace cv;
using namespace std;

struct HogGrabCutTest : ::testing::Test {
  public:
    virtual void SetUp() {
      img = imread("./test.png", CV_LOAD_IMAGE_COLOR);
      param.inputImg = img;
    }
    virtual void TearDown() {}

  public:
    HogGrabCutParam param;
    Mat img;
};

TEST_F(HogGrabCutTest, ShowHogTest) {
  Rect roi1;
  roi1 = Rect(0, 0, img.cols/2, img.rows/2);
  HogGrabCut::ShowHog(img, vector<cv::Rect>{roi1});
  Rect roi2 = Rect(0, 0, img.cols, img.rows);
  HogGrabCut::ShowHog(img, vector<cv::Rect>{roi1, roi2});
}

TEST_F(HogGrabCutTest, ShowImageTest) {
  HogGrabCut::ShowImage(img);
}

TEST_F(HogGrabCutTest, EvalHogStepTest) {
  std::vector<cv::Rect> rects;
  HogGrabCut grabcut(param);
  grabcut.EvalHogStep(rects);
  if (rects.empty()) EXPECT_TRUE(false);
  HogGrabCut::ShowHog(param.inputImg, rects);
}

TEST_F(HogGrabCutTest, EvalGrabCutStepTest) {
  std::vector<cv::Rect> rects;
  HogGrabCut grabcut(param);
  grabcut.EvalHogStep(rects);
  cv::Mat result;
  cv::Mat mask;
  grabcut.EvalGrabCutStep(result, mask, rects);
  std::cout << "result" << std::endl;
  HogGrabCut::ShowGrabCut(result);
  param.inputImg.copyTo(result, mask);
  HogGrabCut::ShowImage(result);
  // HogGrabCut::ShowGrabCut(mask);
}

TEST_F(HogGrabCutTest, EvalTest) {
}
