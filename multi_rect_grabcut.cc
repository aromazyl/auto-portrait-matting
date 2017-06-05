/*
 * multi_rect_grabcut.cc
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <math.h>
#include "multi_rect_grabcut.h"
#include "gmm.h"

using namespace gmm;
using namespace cv;

void MultiRectGrabCut::SetMaskByRect(const std::vector<cv::Rect>& rects) {
  gmm::BuildMaskByRectVec(rects, this->img_.size(), this->mask_);
}

void MultiRectGrabCut::UpdateByKMeans() {
  const int kMeansItCount = 10;
  const int kMeansType = KMEANS_PP_CENTERS;
  Mat bgdLabels, fgdLabels;
  std::vector<Vec3f> bgdSamples, fgdSamples;
  Point p;
  for( p.y = 0; p.y < img_.rows; p.y++ ) {
    for( p.x = 0; p.x < img_.cols; p.x++ ) {
      if( mask_.at<uchar>(p) == GC_BGD || mask_.at<uchar>(p) == GC_PR_BGD )
        bgdSamples.push_back( (Vec3f)img_.at<Vec3b>(p) );
      else // GC_FGD | GC_PR_FGD
        fgdSamples.push_back( (Vec3f)img_.at<Vec3b>(p) );
    }
  }
  CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
  Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
  kmeans( _bgdSamples, frontGroundGMM_.componentsCount, bgdLabels,
      TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
  Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
  kmeans( _fgdSamples, frontGroundGMM_.componentsCount, fgdLabels,
      TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
  backGroundGMM_.initLearning();
  for( int i = 0; i < (int)bgdSamples.size(); i++ )
    backGroundGMM_.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
  backGroundGMM_.endLearning();

  frontGroundGMM_.initLearning();
  for( int i = 0; i < (int)fgdSamples.size(); i++ )
    frontGroundGMM_.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
  frontGroundGMM_.endLearning();
  const double gamma = 50;
  const double lambda = 9*gamma;
  const double beta = calcBeta(img_);
  calcNWeights(img_, leftW_, upleftW_, upW_, uprightW_, beta, gamma);
}

void MultiRectGrabCut::UpdateByGMMTrain() {
  Mat compIdxs( img_.size(), CV_32SC1 );
  this->AssignGMMsComponents(compIdxs);
  this->LearnGMMs(compIdxs);
}

void MultiRectGrabCut::AssignGMMsComponents(Mat& compIdxs) {
  Point p;
  for (p.y = 0; p.y < img_.rows; ++p.y)
    for (p.x = 0; p.x < img_.cols; ++p.x) {

      Vec3d color = img_.at<Vec3b>(p);

      compIdxs.at<int>(p) = mask_.at<uchar>(p) == GC_BGD
        || mask_.at<uchar>(p) == GC_PR_BGD ?
        backGroundGMM_.whichComponent(color)
        : frontGroundGMM_.whichComponent(color);
    }
}

void MultiRectGrabCut::LearnGMMs(Mat& compIdxs) {
  backGroundGMM_.initLearning();
  frontGroundGMM_.initLearning();
  Point p;
  for (int ci = 0; ci < frontGroundGMM_.componentsCount; ci++) {
    for (p.y = 0; p.y < img_.rows; p.y++) {
      for (p.x = 0; p.x < img_.cols; p.x++) {
        if (compIdxs.at<int>(p) == ci ) {
          if (mask_.at<uchar>(p) == GC_BGD || mask_.at<uchar>(p) == GC_PR_BGD )
            backGroundGMM_.addSample(ci, img_.at<Vec3b>(p));
          else
            frontGroundGMM_.addSample(ci, img_.at<Vec3b>(p));
        }
      }
    }
  }
  backGroundGMM_.endLearning();
  frontGroundGMM_.endLearning();
}
void MultiRectGrabCut::UpdateByGrabCut() {
  GCGraph<double> graph;
  this->ConstructGCGraph(graph);
  this->EstimateSegmentation(graph);
}

void MultiRectGrabCut::ConstructGCGraph(GCGraph<double>& graph) {

    constexpr double gamma = 50;
    constexpr double lambda = 9*gamma;
    int vtxCount = img_.cols*img_.rows,
        edgeCount = 2*(4*img_.cols*img_.rows - 3*(img_.cols + img_.rows) + 2);
    graph.create(vtxCount, edgeCount);
    Point p;
    for( p.y = 0; p.y < img_.rows; p.y++ ) {
      for( p.x = 0; p.x < img_.cols; p.x++) {
        // add node
        int vtxIdx = graph.addVtx();
        Vec3b color = img_.at<Vec3b>(p);

        // set t-weights
        double fromSource, toSink;
        if( mask_.at<uchar>(p) == GC_PR_BGD || mask_.at<uchar>(p) == GC_PR_FGD )
        {
          fromSource = -log( backGroundGMM_(color) );
          toSink = -log(frontGroundGMM_(color));
        }
        else if( mask_.at<uchar>(p) == GC_BGD )
        {
          fromSource = 0;
          toSink = lambda;
        }
        else // GC_FGD
        {
          fromSource = lambda;
          toSink = 0;
        }
        graph.addTermWeights( vtxIdx, fromSource, toSink );

        // set n-weights
        if( p.x>0 )
        {
          double w = leftW_.at<double>(p);
          graph.addEdges( vtxIdx, vtxIdx-1, w, w );
        }
        if( p.x>0 && p.y>0 )
        {
          double w = upleftW_.at<double>(p);
          graph.addEdges( vtxIdx, vtxIdx-img_.cols-1, w, w );
        }
        if( p.y>0 )
        {
          double w = upW_.at<double>(p);
          graph.addEdges( vtxIdx, vtxIdx-img_.cols, w, w );
        }
        if( p.x<img_.cols-1 && p.y>0 )
        {
          double w = uprightW_.at<double>(p);
          graph.addEdges( vtxIdx, vtxIdx-img_.cols+1, w, w );
        }
      }
    }
}

void MultiRectGrabCut::EstimateSegmentation(GCGraph<double>& graph) {
  graph.maxFlow();
  Point p;
  for (p.y = 0; p.y < mask_.rows; p.y++) {
    for (p.x = 0; p.x < mask_.cols; p.x++) {
      if (mask_.at<uchar>(p) == GC_PR_BGD || mask_.at<uchar>(p) == GC_PR_FGD) {
        if(graph.inSourceSegment( p.y*mask_.cols+p.x /*vertex index*/ ))
          mask_.at<uchar>(p) = GC_PR_FGD;
        else
          mask_.at<uchar>(p) = GC_PR_BGD;
      }
    }
  }
}

void MultiRectGrabCut::TrainLoopsStart() {
  this->UpdateByKMeans();
  for (int i = 0; i < 5; ++i) {
    this->UpdateByGMMTrain();
    this->UpdateByGrabCut();
  }
}
