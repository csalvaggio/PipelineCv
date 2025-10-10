#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "pcv/PipelineCv.h"

int main() {
  // Read in and validate the original source image
  std::string filename = "../data/images/image.jpg";
  cv::Mat src = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  if (src.empty()) {
    std::cerr << "Failed to read image: " << filename << std::endl;
    return EXIT_FAILURE;
  }

  /*************** FORWARD/INVERSE FOURIER TRANSFORM EXAMPLE ***************/

  // Compute the complex-valued Fourier spectrum
  cv::Mat spectrum =
    src |
    pcv::to_double() |
    pcv::pad_to_even() |
    pcv::dft();

  // Compute the properly-scaled inverse Fourier transform (recovered image)
  cv::Mat recovered =
    spectrum |
    pcv::idft() |
    pcv::to_uint8();

  // Display the workflow images
  cv::imshow("Source",
      src);
  cv::imshow("log |DFT|",
      spectrum |
      pcv::magnitude() |
      pcv::log() |
      pcv::fft_shift() |
      pcv::normalize());
  cv::imshow("Phase(DFT) [-pi to pi)",
      spectrum |
      pcv::phase() |
      pcv::fft_shift() |
      pcv::normalize());
  cv::imshow("Recovered",
      recovered);
  cv::waitKey(0);

  filename = "/home/cnspci/src/cpp/rit/data/images/misc/lenna_grayscale.pgm";
  src = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  if (src.empty()) {
    std::cerr << "Failed to read image: " << filename << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
