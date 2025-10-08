# PipelineCv

A pipeline OpenCV adapter library providing functional-style image processing utilities built on OpenCV.

## Overview

PipelineCv is a tiny, header-only toolkit for composing OpenCV `cv::Mat` transforms using a functional pipe style. Chain operations together for readable, declarative image processing pipelines.

```cpp
cv::Mat img = cv::imread("in.png", cv::IMREAD_COLOR);
 
cv::Mat spectrum = img
  | pcv::to_float()         // convert to CV_32F in [0,1]
  | pcv::to_gray()          // grayscale
  | pcv::pad_to_even()      // FFT-friendly sizes
  | pcv::dft()              // two-channel complex DFT
  | pcv::fft_shift()        // center DC
  | pcv::power_spectrum();  // |F|^2 (log)
```

## Requirements

- C++ compiler that supports C++20 dialect/ISO standard
- OpenCV (v4.3.0 or higher) (core/imgproc)

## Installation

PipelineCv is header-only. Simply include it in your project (for example):

```cpp
#include "pcv/PipelineCv.h"
```

## Documentation

See the generated Doxygen documentation for complete [API reference](https://home.cis.rit.edu/~cnspci/other/apis/pcv/).

## Error Handling

Most adapters use `CV_Assert` for precondition checks. For example, complex input operations validate that the input has exactly 2 channels.

## Performance Tips

- Prefer `CV_32F` for frequency operations—convert once, process, then convert back
- Use `pcv::pad_to_even()` before `pcv::dft()` for clean quadrant swaps
- Compose operations to minimize intermediate copies when practical

## Quick Examples

### Normalize
```cpp
cv::Mat out = img 
  | pcv::to_float() 
  | pcv::normalize(0.0, 1.0);
```

### Fourier Transform & Magnitude
```cpp
cv::Mat mag = img 
  | pcv::to_float() 
  | pcv::to_gray() 
  | pcv::pad_to_even()
  | pcv::to_complex() 
  | pcv::dft() 
  | pcv::magnitude();
```

### Image Blend
```cpp
cv::Mat blended = img 
  | pcv::add(other, alpha=0.7, beta=0.3, gamma=0.0)
  | pcv::clamp(0, 255);
```

## Notes

- The pipe operator accepts any callable with signature `cv::Mat(const cv::Mat&)`
- Complex operations require 2-channel inputs (`CV_32FC2` or `CV_64FC2`)
- For FFT visualization, use even dimensions (`pcv::pad_to_even()`) and center DC with `pcv::fft_shift()`

&nbsp;
## Contact

**Carl Salvaggio, Ph.D.**  
Email: carl.salvaggio@rit.edu

Chester F. Carlson Center for Imaging Science  
Rochester Institute of Technology  
Rochester, New York 14623  
United States
