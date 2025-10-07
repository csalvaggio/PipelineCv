# PipelineCv - A pipeline OpenCV adapter library

### DESCRIPTION
The PipelineCv defined adapters provide a tiny, header-only toolkit for composing OpenCV cv::Mat transforms using a functional pipe style:

```
cv::Mat img = cv::imread("in.png", cv::IMREAD_COLOR);
 
cv::Mat spectrum = img
  | pcv::to_float()         // convert to CV_32F in [0,1]
  | pcv::to_gray()          // grayscale
  | pcv::pad_to_even()      // FFT-friendly sizes
  | pcv::dft()              // two-channel complex DFT
  | pcv::fft_shift()        // center DC
  | pcv::power_spectrum();  // |F|^2 (log)
```

### BINARIES
Binaries (executables) will be located within the `build/bin` directory.

### REQUIREMENTS
* C++ compiler that supports C++20 dialect/ISO standard

### DEPENDENCIES 
* OpenCV 3 (v. 4.3.0 or higher)
    
### TESTING 
In the build directory
```
bin/test_pipeline_cv
```

&nbsp;
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