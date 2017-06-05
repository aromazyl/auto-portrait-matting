/*
 * hog_grabcut.cc
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "hog_grabcut.h"

HogGrabCut::HogGrabCut(
    const HogGrabCutParam& param,
    const std::vector<float>& descriptor) {

  this->UpdateParam(param);

  this->grab_cut_.UpdateImage(param.inputImg);

  if (descriptor.empty()) {
    hogDescriptor_.setSVMDetector(
        cv::HOGDescriptor::getDefaultPeopleDetector());
  } else {
    hogDescriptor_.setSVMDetector(descriptor);
  }
};

HogGrabCut::~HogGrabCut() {}

void HogGrabCut::Eval(cv::Mat& result, cv::Mat& mask, std::vector<cv::Rect>& rects) {
  this->EvalHogStep(rects);

  if (this->param_.ShowImage)
    this->ShowHog(this->param_.inputImg, rects);

  this->EvalGrabCutStep(result, mask, rects);

  if (this->param_.ShowImage)
    this->ShowGrabCut(result);
}


void HogGrabCut::EvalHogStep(std::vector<cv::Rect>& rects) {
  std::vector<cv::Rect> found;

  hogDescriptor_.detectMultiScale(
      this->param_.inputImg,
      found,
      this->param_.MultiScale_scaleFactor,
      this->param_.MultiScale_minNeiborhoods,
      this->param_.MultiScale_flag,
      this->param_.MultiScale_min_cvSize,
      this->param_.MultiScale_max_cvSize);

  for(size_t i = 0; i < found.size(); i++ ) {
    cv::Rect& r = found[i];
    size_t j;
    for ( j = 0; j < found.size(); j++ )
      if ( j != i && (r & found[j]) == r )
        break;

    if ( j == found.size() )
      rects.push_back(r);
  }

  for (auto& r : rects) {
    r.x += cvRound(r.width*0.1);
    r.width = cvRound(r.width*0.8);
    r.y += cvRound(r.height*0.07);
    r.height = cvRound(r.height*0.8);
  }
}

void HogGrabCut::EvalGrabCutStep(cv::Mat& pic_result, cv::Mat& mask_result, const std::vector<cv::Rect>& rects) {
  this->grab_cut_.SetMaskByRect(rects);
  this->grab_cut_.TrainLoopsStart();
  mask_result = this->grab_cut_.GetFinalMask();
  mask_result = mask_result & 1;
  this->param_.inputImg.copyTo(pic_result, mask_result);
}

void HogGrabCut::ShowHog(const cv::Mat& img, const std::vector<cv::Rect>& rects) {
  if (rects.empty()) return;

  for (auto& rect : rects) {
    HogGrabCut::ShowImage(img(rect));
  }
}

void HogGrabCut::ShowGrabCut(const cv::Mat& mat) {
  HogGrabCut::ShowImage(mat);
}

void HogGrabCut::ShowImage(const cv::Mat& mat) {
  if (mat.empty()) return;

  for (;;) {
    cv::imshow("", mat);
    int c = cv::waitKey( 30 ) & 255;
    if ( c == 'q' || c == 'Q' || c == 27)
      break;
  }
}

void HogGrabCut::UpdateParam(const HogGrabCutParam& param) {
  this->param_ = param;
}


void HogGrabCut::GetTrimap(const cv::Mat& mapori,
      const cv::Mat& mapconverted, cv::Mat& trimap) {
}
