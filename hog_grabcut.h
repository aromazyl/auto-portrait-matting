/*
 * hog_grabcut.h
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef HOG_GRABCUT_H
#define HOG_GRABCUT_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include "multi_rect_grabcut.h"

struct HogGrabCutParam {
  cv::Mat inputImg;
  double MultiScale_scaleFactor = 0;
  cv::Size MultiScale_minNeiborhoods = cv::Size(8, 8);
  cv::Size MultiScale_flag = cv::Size(32, 32);
  double MultiScale_min_cvSize = 1.05;
  double MultiScale_max_cvSize = 2;
  bool ShowImage = false;
};

class HogGrabCut {
public:
  static void ShowHog(const cv::Mat&, const std::vector<cv::Rect>&);
  static void ShowGrabCut(const cv::Mat&);
  static void ShowImage(const cv::Mat&);
  static void GetTrimap(const cv::Mat& mapori,
      const cv::Mat& mapconverted, cv::Mat& trimap);

public:
  explicit HogGrabCut(
      const HogGrabCutParam&,
      const std::vector<float>& desc = std::vector<float>{});
  ~HogGrabCut();

public:
  void UpdateParam(const HogGrabCutParam& param);

#ifdef GTEST
public:
#else
private:
#endif
  void EvalHogStep(std::vector<cv::Rect>& rects);
  void EvalGrabCutStep(cv::Mat& pic_result, cv::Mat& mask_result, const std::vector<cv::Rect>& rects);

public:
  void Eval(cv::Mat& result, cv::Mat& mask, std::vector<cv::Rect>& rectangle);

#ifdef GTEST
public:
#else
private:
#endif

  cv::HOGDescriptor hogDescriptor_;
  HogGrabCutParam param_;
  MultiRectGrabCut grab_cut_;
};

#endif /* !HOG_GRABCUT_H */
