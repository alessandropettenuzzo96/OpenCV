//
//  OpenCV.mm
//
//  Created by Danny on 14/11/2019.
//  Copyright © 2019 Danny Zorin. All rights reserved.
//

#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>

#import "OpenCVWrapper.h"
#import "UIImage+OpenCV.h"

using namespace std;
using namespace cv;

@implementation OpenCVWrapper

#pragma mark - Saliency on image

// https://docs.opencv.org/trunk/d4/dee/tutorial_optical_flow.html

// When using 3D Camera for taking 3d photos,
// We need to find one focus point for all images
// For that we need to track points movements and select best point
// Then we use these movement info to translate and crop images
// So all images will be aligned by one point

+ (NSArray<NSValue*>*)findOpticalFlowMovement:(NSArray<UIImage*>*)images focusPoint:(CGPoint)focusPoint pointType:(CVFocusPointType)pointType faceRect:(CGRect)faceRect {

  int n_frames = int(images.count);
  
  vector <Point2f> prev_pts, curr_pts;
  
  Mat curr, curr_gray;
  Mat prev, prev_gray;
  
  prev = images[0].CVMat;
  cvtColor(prev, prev_gray, COLOR_BGR2GRAY);
  
  Mat searchMask = Mat::zeros(prev.size(), CV_8U);
  float width = prev.size().width;
  float height = prev.size().height;
  float maskX = 0, maskY = 0, maskW = 0, maskH = 0;

  int numPoints = 430;
  int win = 53;
  if (pointType == CVFocusPointTypeOpenCV) {
    maskX = width * 0.27;
    maskY = height * 0.16;
    maskW = width - maskX * 2;
    maskH = height - maskY - height * 0.35;
  } else if (pointType == CVFocusPointTypeFace) {
    bool isFrontCameraFace = (faceRect.size.width / width) > 0.38;
    if (isFrontCameraFace) {
      numPoints = 800;
      win = 71;
    }
    float wMultiplier = isFrontCameraFace ? -0.15 : 0.12;
    float hMultiplier = isFrontCameraFace ? -0.05 : 0.12;
    float wOffset = faceRect.size.width * wMultiplier;
    float hOffset = faceRect.size.height * hMultiplier;
    maskX = MAX(0, faceRect.origin.x - wOffset / 2);
    maskY = MAX(0, faceRect.origin.y - hOffset / 2);
    maskW = MIN(faceRect.size.width + wOffset, width - maskX);
    maskH = MIN(faceRect.size.height + hOffset, height - maskY);
  } else if (pointType == CVFocusPointTypeManual) {
    maskX = MAX(0, focusPoint.x - width * 0.15);
    maskY = MAX(0, focusPoint.y - height * 0.15);
    maskW = MIN(width * 0.3, width - maskX);
    maskH = MIN(height * 0.3, height - maskY);
  }
  
  Mat roi(searchMask, cv::Rect(maskX,maskY,maskW,maskH));
  roi = Scalar(255);
  
  goodFeaturesToTrack(prev_gray, prev_pts, numPoints, 0.01, 11, searchMask, 7, false, 0.04);
  
  int minStartPoints = pointType == CVFocusPointTypeFace ? 40 : 60;
  if (prev_pts.size() < minStartPoints) {
    return [NSArray array];
  }
  
  // First iteration, fill all found tracks
  vector<vector<Point2f>> tracks;
  for(int i = 0; i < prev_pts.size(); i++) {
    vector<Point2f> newTrack;
    newTrack.push_back(prev_pts[i]);
    tracks.push_back(newTrack);
  }
    
  // For debug purposes
//   vector<Scalar> colors;
//   RNG rng;
//   for(int i = 0; i < prev_pts.size(); i++) {
//     int r = rng.uniform(0, 256);
//     int g = rng.uniform(0, 256);
//     int b = rng.uniform(0, 256);
//     colors.push_back(Scalar(r,g,b));
//   }
    
  TermCriteria termcrit(TermCriteria::COUNT+TermCriteria::EPS, 120, 0.005);
  cv::Size winSize(win, win);
  int maxLevel = 5;
  
  for (int i = 1; i < n_frames; i++) {
    @autoreleasepool {
      
      curr = images[i].CVMat;
      cvtColor(curr, curr_gray, COLOR_BGR2GRAY);
            
      vector <uchar> status;
      vector <float> err;
      calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err, winSize, maxLevel, termcrit, 0, 0.0001);
      
      vector<Point2f> points_back;
      vector<unsigned char> status_back;
      vector<float> err_back;
      calcOpticalFlowPyrLK(curr_gray, prev_gray, curr_pts, points_back, status_back, err_back, winSize, maxLevel, termcrit, 0, 0.0001);
      
      // Traverse vector backward so we can remove points on the fly
      float thr_fb = 0.2;
      for (int i = int(prev_pts.size()) - 1; i >= 0; i--) {
        float l2norm = norm(points_back[i] - prev_pts[i]);
        bool fb_err_is_large = l2norm > thr_fb;
        if (fb_err_is_large || !status[i] || !status_back[i]) {
          status[i] = 0;
        }
      }
            
      // Filter only valid points
      auto prev_it = prev_pts.begin();
      auto curr_it = curr_pts.begin();
      auto tracks_it = tracks.begin();
//      auto colors_it = colors.begin(); // For debug
      for (size_t k = 0; k < status.size(); k++) {
        if(status[k]) {
          prev_it++;
          curr_it++;
          tracks_it++;
//          colors_it++; // For debug

          tracks[k].push_back(curr_pts[k]);
        } else {
          prev_it = prev_pts.erase(prev_it);
          curr_it = curr_pts.erase(curr_it);
          tracks_it = tracks.erase(tracks_it);
//          colors_it = colors.erase(colors_it); // For debug
        }
      }
      
      int minIterationPoints = pointType == CVFocusPointTypeFace ? 10 : 20;
      if (curr_pts.size() < minIterationPoints) {
        return [NSArray array];
      }
      
      prev_gray = curr_gray.clone();
      prev_pts = curr_pts;
    }
  }
  
  vector<vector<Point2f>> goodTracks;
  for(int m = 0; m < tracks.size(); m++) {
    vector<Point2f> trackPoints = tracks[m];
    if (trackPoints.size() >= n_frames) {
      goodTracks.push_back(trackPoints);
    }
  }
  
  int minGoodTrackPoints = pointType == CVFocusPointTypeFace ? 3 : 5;
  if (goodTracks.size() < minGoodTrackPoints) {
    return [NSArray array];
  }
  
//  NSLog(@"Found GOOD Tracks size: %lu", goodTracks.size());
  
  // For debug purposes
//  for(int i = 0; i < goodTracks.size(); i++) {
//    vector<Point2f> trackPoints = goodTracks[i];
//
//    double distSum = 0;
//    for(int k = 0; k < trackPoints.size(); k++) {
//      if (k >= 1) {
//        float dist = sqrt(norm(trackPoints[k] - trackPoints[k-1]));
//        distSum += dist;
//      }
//    }
//    NSLog(@"(%d) Distance: %f", i, distSum);
//  }

  // For debug purposes
//  Mat testMask = Mat::zeros(prev.size(), prev.type());
//  saveImageToDisk(images[0].CVMat);
//  for (int i = 1; i < n_frames; i++) {
//    @autoreleasepool {
//      curr = images[i].CVMat;
//
//      for(int m = 0; m < goodTracks.size(); m++) {
//        vector<Point2f> trackPoints = goodTracks[m];
//        Scalar trackColor = colors[m];
//
//        line(testMask, trackPoints[i-1], trackPoints[i], trackColor, 2);
//        if (n_frames - 1 == i) {
//          circle(curr, trackPoints[i], 5, trackColor, -1);
//        }
//      }
//
//      Mat img;
//      add(curr, testMask, img);
//      saveImageToDisk(img);
//    }
//  }
  
  // Find closest track point to focus point
  vector<Point2f> cloud2d;
  for(int i = 0; i < goodTracks.size(); i++) {
    vector<Point2f> trackPoints = goodTracks[i];
    cloud2d.push_back(trackPoints[0]);
  }
  
  Point2f sourcePoint = Point2f(focusPoint.x, focusPoint.y);;
  flann::KDTreeIndexParams indexParams(4);
  flann::Index kdtree(Mat(cloud2d).reshape(1), indexParams);
  
  vector<float> query;
  query.push_back(sourcePoint.x);
  query.push_back(sourcePoint.y);
  vector<int> indices;
  vector<float> dists;
  kdtree.knnSearch(query, indices, dists, 1);

  vector<Point2f> bestTrack;
  for(int k = 0; k < indices.size(); k++) {
    int cloudIndex = indices[k];
    if (dists[k] > 0.0) {
      bestTrack = goodTracks[cloudIndex];
      
//      Mat testImage = images[0].CVMat;
//      line(testImage, bestTrack[0], sourcePoint, Scalar(0,255,0), 2, LINE_AA); // for debug
//      saveImageToDisk(testImage);
      //NSLog(@"point at index: %d (%f,%f) is on distance - %f", cloudIndex, focusPoint.x, focusPoint.y, sqrt(dists[k]));
    }
  }
  
  NSMutableArray* trackPoints = [NSMutableArray array];
  for(int i = 0; i < bestTrack.size(); i++) {
    CGPoint cgPoint = CGPointMake(bestTrack[i].x, bestTrack[i].y);
    [NSValue valueWithCGPoint:cgPoint];
    [trackPoints addObject:[NSValue valueWithCGPoint:cgPoint]];
  }
  
  return trackPoints;
}

+ (CGPoint)getCenterPointOnImage:(UIImage*)image {
  Point2f centerPoint = Point2f(image.size.width * 0.5, image.size.height * 0.48);
  return CGPointMake(centerPoint.x, centerPoint.y);
}

void saveImageToDisk(Mat cvImage) {
  UIImage *uiImage = [[UIImage alloc] initWithCVMat:cvImage];
  UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil);
}

// This function checks how much of foreground is on the image.
// The more foreground means that object is closer to camera
// Less foreground means object is somewhere on center and we need more rotation for 3d photo
+ (NSNumber*)processDepthImage:(UIImage*)depthImage isPredicted:(BOOL)isPredicted {

  Mat depthIm = depthImage.CVMat;
  Mat depthImGray;
  cvtColor(depthIm, depthImGray, COLOR_BGR2GRAY);
    
  Mat normalizedDepth;
  normalize(depthImGray, normalizedDepth, 0, 255, NORM_MINMAX);

  // Invert needed because depth has black as near area,
  // But we need white to be near
  Mat invertedDepth;
  bitwise_not(normalizedDepth, invertedDepth);
  
  Mat thresholdedFocus;
  double thresh = isPredicted ? 196 : 200; // was 180 for predicted
  threshold(invertedDepth, thresholdedFocus, thresh, 255, THRESH_BINARY);
  
  float width = depthIm.size().width;
  float height = depthIm.size().height;
  Mat mask = Mat::zeros(depthIm.size(), CV_8U);
  mask(cv::Rect(width * 0.12, height * 0.12, width * 0.76, height * 0.76)) = 1;
  Mat thresholdedFocusMasked;
  thresholdedFocus.copyTo(thresholdedFocusMasked, mask);
    
  // Find contours to use as new edges
  vector<vector<cv::Point>> focusContours;
  vector<Vec4i> focusHierarchy;
  findContours(thresholdedFocusMasked, focusContours, focusHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  
  //NSLog(@"Found Focus Countours count %lu", focusContours.size());
  
  // Contour Area
  double fgArea = 0;
  for(unsigned i = 0; i < focusContours.size(); i++) {
    double area = sqrt(contourArea(focusContours[i]));
    if (isPredicted) {
      fgArea = max(area, fgArea);
    } else {
      fgArea += area;
    }
  }
  
  
  CGFloat d3dFocus = 0.5;
  if (isPredicted) {
    double fullArea = sqrt((width * 0.76) * (height * 0.76));
    double fgPercentage = fgArea / fullArea;
    //NSLog(@"fullArea: %f | percent fg: %f", fullArea, (fgArea / fullArea));
    if (fgPercentage < 0.2) {
      d3dFocus = 0.32;
    } else if (fgPercentage < 0.34) {
      d3dFocus = 0.34;
    } else if (fgPercentage < 0.38) {
      d3dFocus = 0.38;
    } else if (fgPercentage < 0.42) {
      d3dFocus = 0.44;
    }
  } else {
    if (fgArea < 50) {
      d3dFocus = 0.32;
    } else if (fgArea < 400) {
      d3dFocus = 0.36;
    } else if (fgArea < 600) {
      d3dFocus = 0.45;
    }
  }
  
  //NSLog(@"contour Area Sum is %f. Focus: %f", fgArea, d3dFocus);
  
  return @(d3dFocus);
}

// Checking how much background presented on the image
// Background depth values may be higher when placing sticker
// For that I analyze image background and find better threshold for sticker filter
+ (NSNumber*)findBackgroundAmount:(UIImage*)depthImage isPredicted:(BOOL)isPredicted defaultThresh:(CGFloat)defaultThresh {
  
  Mat depthIm = depthImage.CVMat;
  Mat depthImGray;
  cvtColor(depthIm, depthImGray, COLOR_BGR2GRAY);
  
  // Inversion needed because disparity has black as background area,
  // But we need white to be background
  Mat invertedDepth;
  bitwise_not(depthImGray, invertedDepth);
  
  //saveImageToDisk(invertedDepth); // For debug

  Mat thresholdedFocus;
  double thresh = isPredicted ? 198 : 220;
  threshold(invertedDepth, thresholdedFocus, thresh, 255, THRESH_BINARY);
  
  float width = depthIm.size().width;
  float height = depthIm.size().height;
  Mat mask = Mat::zeros(depthIm.size(), CV_8U);
  mask(cv::Rect(0.0, 0.0, width, height * 0.6)) = 1;
  Mat thresholdedFocusMasked;
  thresholdedFocus.copyTo(thresholdedFocusMasked, mask);
    
  //saveImageToDisk(thresholdedFocusMasked); // For debug 
    
  vector<vector<cv::Point>> focusContours;
  vector<Vec4i> focusHierarchy;
  findContours(thresholdedFocusMasked, focusContours, focusHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
  // Contour Area
  double bgArea = 0;
  for(unsigned i = 0; i < focusContours.size(); i++) {
    bgArea += contourArea(focusContours[i]);
  }

  CGFloat stickerBgThresh = defaultThresh;
  double fullArea = (width) * (height * 0.6);
  double bgPercentage = bgArea / fullArea;
  
  if (isPredicted) {
    if (bgPercentage < 0.04) {
      stickerBgThresh = 0.44;
    } else if (bgPercentage < 0.2) {
      stickerBgThresh = 0.4;
    } else if (bgPercentage < 0.3) {
      stickerBgThresh = 0.38;
    } else if (bgPercentage < 0.45) {
      stickerBgThresh = 0.35;
    } else if (bgPercentage > 0.8) {
      stickerBgThresh = 0.24;
    } 
  } else {
    if (bgPercentage < 0.1) {
      stickerBgThresh = 0.36;
    } else if (bgPercentage < 0.25) {
      stickerBgThresh = 0.26;
    } else if (bgPercentage > 0.85) {
      stickerBgThresh = 0.1;
    }
  }
  
  // NSLog(@"fullArea: %f | percent bg: %f", fullArea, (bgArea / fullArea));
  // NSLog(@"contour Area Sum is %f. StickerBgThresh: %f", bgArea, stickerBgThresh);
  
  return @(stickerBgThresh);
}

@end
