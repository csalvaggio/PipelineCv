# PipelineCv - A pipeline OpenCV adapter library {#mainpage}

## DESCRIPTION
The PipelineCv defined adapters provide a tiny, header-only toolkit for composing OpenCV cv::Mat transforms using a functional pipe style:

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

## BUILD & INCLUDE
- Requires a C++ compiler that supports C++20 dialect/ISO standard
- Requires OpenCV 3 (v. 4.3.0 or higher) (core/imgproc)
- Include the header where you build your pipeline (here) ...

```cpp
#include "pcv/PipelineCv.h"
```

## TESTING 
In the build directory
```bash
bin/test_pipeline_cv
```

## QUICK EXAMPLES

### Normalize and Clamp
```cpp
cv::Mat out = img | pcv::to_float() | pcv::normalize(0.0, 1.0) | pcv::clamp(0.0, 1.0);
```

### Fourier Transform & Magnitude
```cpp
cv::Mat mag = img | pcv::to_float() | pcv::to_gray() | pcv::to_complex() | pcv::dft() | pcv::magnitude();
```

### Affine Blend
```cpp
cv::Mat blended = img | pcv::add(other, alpha=1.0, beta=0.7, gamma=0.0);
```

## API REFERENCE

See the organized function groups:
- @ref type_and_channel_adapters - Type conversions and channel extraction/combination
- @ref scalar_value_transforms - Scalar/value transformations and per-pixel operations
- @ref elementwise_operations - Element-wise arithmetic between two cv::Mats
- @ref complex_domain_operations - Complex-domain operations
- @ref fourier_frequency_domain_utilities - Fourier and frequency domain utilities
- @ref spatial_utilities - Spatial/geometric modifications

## NOTES
- The pipe operator accepts any callable with signature `cv::Mat(const cv::Mat&)`
- Helpers that expect complex data require 2-channel inputs (`CV_32FC2`/`CV_64FC2`)
- For FFT visualization, prefer even dimensions (pcv::pad_to_even()) and center DC with pcv::fft_shift()

## ERROR HANDLING
- Most adapters use `CV_Assert` for precondition checks (e.g., complex inputs must have 2 channels)

## PERFORMANCE TIPS
- Prefer `CV_32F` for frequency operations; convert once, process, then convert back
- Use pcv::pad_to_even() before pcv::dft() for clean quadrant swaps with pcv::fft_shift()
- Compose operations to minimize intermediate copies when practical

---

# Contact
### Author  
Carl Salvaggio, Ph.D.

### E-mail
carl.salvaggio@rit.edu

### Organization
Chester F. Carlson Center for Imaging Science  
Rochester Institute of Technology  
Rochester, New York, 14623  
United States
