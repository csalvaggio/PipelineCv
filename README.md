# PipelineCv

Tiny, composable pipelines for OpenCV images using a readable, functional style.
This README doubles as the **Doxygen main page**.

> Tip: In the generated HTML docs, this page is the landing page. See also the
> API overview page: \ref pipelinecv_overview.

---

## Quick Start

### Requirements
- C++20 compiler
- CMake ≥ 3.20
- OpenCV (4.x recommended)

### Configure & Build (example)
```bash
# from your project root
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Generate Documentation
```bash
doxygen Doxyfile
# Open docs/html/index.html in your browser
```

---

## Usage (example)

```cpp
#include <opencv2/imgcodecs.hpp>
#include <pcv/PipelineCv.h>

int main() {
  cv::Mat img = cv::imread("in.png", cv::IMREAD_COLOR);

  cv::Mat spectrum = img
    | pcv::to_float()        // convert to CV_32F [0,1]
    | pcv::to_gray()         // grayscale
    | pcv::pad_to_even()     // FFT-friendly sizes
    | pcv::dft()             // two-channel complex DFT
    | pcv::fft_shift()       // center DC
    | pcv::power_spectrum(); // |F|^2 (log)

  cv::imwrite("spectrum.png", spectrum * 255.0f);
  return 0;
}
```

**Notes**
- For FFT visualization, prefer even dimensions (`pcv::pad_to_even`) and center DC with `pcv::fft_shift`.
- When chaining, ensure each adapter’s input type matches its expectation.

---

## Project Layout (suggested)
```
.
├── CMakeLists.txt
├── include/
│   └── pcv/
│       └── PipelineCv.h
├── src/ (optional)
├── tests/ (optional)
├── README.md          # ← main page for Doxygen
├── PipelineCv.dox     # ← API overview / topic page
└── Doxyfile
```

---

## Links
- API Overview: \ref pipelinecv_overview
- OpenCV: https://opencv.org/
