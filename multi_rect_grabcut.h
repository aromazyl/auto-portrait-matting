/*
 * multi_rect_grabcut.h
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef MULTI_RECT_GRABCUT_H
#define MULTI_RECT_GRABCUT_H

#include "gcgraph.hpp"
#include "gmm.h"

class MultiRectGrabCut {
public:
  MultiRectGrabCut() : frontGroundGMM_(gmmF_), backGroundGMM_(gmmB_) {}
  ~MultiRectGrabCut() {}
  void UpdateImage(const cv::Mat& img) { this->img_ = img; }
  void SetMaskByRect(const std::vector<cv::Rect>& rects);
  void TrainLoopsStart();
  cv::Mat& GetFinalMask() { return mask_; }

#ifdef GTEST
private:
#else
public:
#endif
  void UpdateByKMeans();
  void UpdateByGMMTrain();
  void UpdateByGrabCut();


  void AssignGMMsComponents(cv::Mat& compIdxs);
  void LearnGMMs(cv::Mat& compIdxs);

  void ConstructGCGraph(GCGraph<double>& graph);
  void EstimateSegmentation(GCGraph<double>& graph);

private:
  cv::Mat img_;
  cv::Size imgSize_;
  cv::Mat mask_;
  cv::Mat gmmF_;
  cv::Mat gmmB_;
  gmm::GMM<> frontGroundGMM_;
  gmm::GMM<> backGroundGMM_;
  cv::Mat leftW_;
  cv::Mat upleftW_;
  cv::Mat upW_;
  cv::Mat uprightW_;
};

#endif /* !MULTI_RECT_GRABCUT_H */
