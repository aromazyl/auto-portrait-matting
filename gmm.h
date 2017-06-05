/*
 * gmm.h
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GMM_H
#define GMM_H


#include <opencv2/opencv.hpp>

namespace gmm {

template <int components_count = 5, int iter_count = 1>
class GMM {
public:
  static const int componentsCount = components_count;
  static const int iterCount = iter_count;

  GMM(cv::Mat& _model);
  double operator()( const cv::Vec3d color ) const;
  double operator()( int ci, const cv::Vec3d color ) const;
  int whichComponent( const cv::Vec3d color ) const;

  void initLearning();
  void addSample( int ci, const cv::Vec3d color );
  void endLearning();

#ifndef GTEST
private:
#else
public:
#endif
  void calcInverseCovAndDeterm(int ci);
  cv::Mat model;
  double* coefs;
  double* mean;
  double* cov;

  double inverseCovs[componentsCount][3][3];
  double covDeterms[componentsCount];

  double sums[componentsCount][3];
  double prods[componentsCount][3][3];
  int sampleCounts[componentsCount];
  int totalSampleCount;

};

#include "gmm.cc"

inline void BuildMaskByRectVec(const std::vector<cv::Rect>& rects,
   const cv::Size& imgSize, cv::Mat& mask) {
  mask.create(imgSize, CV_8UC1);
  mask.setTo(cv::GC_BGD);
  cv::Rect rect_buf;

  for (auto& rect : rects) {
    rect_buf.x = std::max(0, rect.x);
    rect_buf.y = std::max(0, rect.y);
    rect_buf.width = std::min(rect.width, imgSize.width - rect.x);
    rect_buf.height = std::min(rect.height, imgSize.height-rect.y);
    (mask(rect_buf)).setTo( cv::Scalar(cv::GC_PR_FGD) );
  }
}

inline void calcNWeights(const cv::Mat& img, cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW, double beta, double gamma) {
  const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
  leftW.create( img.rows, img.cols, CV_64FC1 );
  upleftW.create( img.rows, img.cols, CV_64FC1 );
  upW.create( img.rows, img.cols, CV_64FC1 );
  uprightW.create( img.rows, img.cols, CV_64FC1 );
  for (int y = 0; y < img.rows; y++) {
    for ( int x = 0; x < img.cols; x++) {
      cv::Vec3d color = img.at<cv::Vec3b>(y,x);
      if( x-1>=0 ) // left
      {
        cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y,x-1);
        leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
      }
      else
        leftW.at<double>(y,x) = 0;
      if( x-1>=0 && y-1>=0 ) // upleft
      {
        cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x-1);
        upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
      }
      else
        upleftW.at<double>(y,x) = 0;
      if( y-1>=0 ) // up
      {
        cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x);
        upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
      }
      else
        upW.at<double>(y,x) = 0;
      if( x+1<img.cols && y-1>=0 ) // upright
      {
        cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x+1);
        uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
      }
      else
        uprightW.at<double>(y,x) = 0;
    }
  }
}

inline double calcBeta( const cv::Mat& img ) {
  double beta = 0;
  for( int y = 0; y < img.rows; y++ )
  {
    for( int x = 0; x < img.cols; x++ )
    {
      cv::Vec3d color = img.at<cv::Vec3b>(y,x);
      if( x>0 ) // left
      {
        cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y,x-1);
        beta += diff.dot(diff);
      }
      if( y>0 && x>0 ) // upleft
      {
        cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x-1);
        beta += diff.dot(diff);
      }
      if( y>0 ) // up
      {
        cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x);
        beta += diff.dot(diff);
      }
      if( y>0 && x<img.cols-1) // upright
      {
        cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y-1,x+1);
        beta += diff.dot(diff);
      }
    }
  }
  if( beta <= std::numeric_limits<double>::epsilon() )
    beta = 0;
  else
    beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

  return beta;
}
}



#endif /* !GMM_H */
