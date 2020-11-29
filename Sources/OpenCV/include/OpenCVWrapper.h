//
//  OpenCV.h
//  
//
//  Created by Danny on 14/11/2019.
//  Copyright © 2019 Danny Zorin. All rights reserved.
//
#include "PrefixHeader.pch"
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, CVFocusPointType) {
    CVFocusPointTypeManual,
    CVFocusPointTypeFace,
    CVFocusPointTypeOpenCV
};

@interface OpenCVWrapper : NSObject

+ (NSArray<NSValue*>*)findOpticalFlowMovement:(NSArray<UIImage*>*)images focusPoint:(CGPoint)focusPoint pointType:(CVFocusPointType)pointType faceRect:(CGRect)faceRect;

+ (CGPoint)getCenterPointOnImage:(UIImage*)image;

+ (NSNumber*)processDepthImage:(UIImage*)depthImage isPredicted:(BOOL)isPredicted;

+ (NSNumber*)findBackgroundAmount:(UIImage*)depthImage isPredicted:(BOOL)isPredicted defaultThresh:(CGFloat)defaultThresh;

@end

NS_ASSUME_NONNULL_END


