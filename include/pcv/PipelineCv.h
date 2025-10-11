#ifndef PIPELINE_CV_H_
#define PIPELINE_CV_H_

/**
 * @file PipelineCV.h
 * @brief Functional-style image processing utilities built on OpenCV.
 *
 * This header defines a minimal "pipe" operator (`|`) and a collection of
 * composable adapters returning `cv::Mat(const cv::Mat&)` callables.
 * They can be chained together for readable, declarative pipelines:
 *
 * @code
 * cv::Mat img = cv::imread("foo.png");
 * cv::Mat result = img
 *   | pcv::to_float()           // convert to 32F in [0,1]
 *   | pcv::to_gray()            // grayscale
 *   | pcv::normalize(0.0, 1.0)  // normalize min->0, max->1
 *   | pcv::multiply_scalar(2.0) // scale
 *   | pcv::clamp(0.0, 1.0);     // clip
 * @endcode
 *
 * @note All functions are defined `inline` for header-only inclusion.
 * @see https://docs.opencv.org/
 *
 * @author Carl Salvaggio
 * @date 2025-10-06
 * @version 1.0
 * @license MIT
 */

#include <functional>
#include <opencv2/opencv.hpp>
#include <vector>

namespace pcv {

/**
 * @brief Pipe operator to compose cv::Mat -> cv::Mat callables.
 *
 * Enables a readable left-to-right flow style.
 *
 * @param in  Input image.
 * @param f   Callable of signature `cv::Mat(const cv::Mat&)`.
 * @return Output of `f(in)`.
 */
inline cv::Mat operator|(const cv::Mat& in,
                         const std::function<cv::Mat(const cv::Mat&)>& f) {
  return f(in);
}

// ============================================================================
// 1) Type and Channel Adapters
// ============================================================================

/**
 * @brief Convert image to 32-bit float (`CV_32F`), with optional scaling.
 *
 * If the input is already `CV_32F`, it's returned unchanged.
 *
 * @param scale Multiplicative factor applied during conversion (default 1/255).
 * @return Callable producing `CV_32F`.
 *
 * \ingroup type_and_channel_adapters
 */
inline auto to_float(double scale = 1.0 / 255.0) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    if (in.type() != CV_32F)
      in.convertTo(out, CV_MAKETYPE(CV_32F, in.channels()), scale);
    else
      out = in;
    return out;
  };
}

/**
 * @brief Convert image to 64-bit float (`CV_64F`), with optional scaling.
 *
 * If the input is already `CV_64F`, it's returned unchanged.
 *
 * @param scale Multiplicative factor applied during conversion (default 1/255).
 * @return Callable: `cv::Mat(const cv::Mat&) -> cv::Mat (CV_64F)`.
 *
 * \ingroup type_and_channel_adapters
 */
inline auto to_double(double scale = 1.0 / 255.0) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    if (in.type() != CV_64F)
      in.convertTo(out, CV_MAKETYPE(CV_64F, in.channels()), scale);
    else
      out = in;
    return out;
  };
}

/**
 * @brief Convert image to 8-bit unsigned (`CV_8U`), with optional scaling.
 *
 * If the input is already `CV_8U`, it's returned unchanged.
 *
 * @param scale Multiplicative factor applied during conversion (default 255).
 * @return Callable producing `CV_8U`.
 *
 * \ingroup type_and_channel_adapters
 */
inline auto to_uint8(double scale = 255.0) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    if (in.type() != CV_8U)
      in.convertTo(out, CV_MAKETYPE(CV_8U, in.channels()), scale);
    else
      out = in;
    return out;
  };
}

/**
 * @brief Convert color or color+annotation to single-channel grayscale.
 *
 * If the input already has one channel, it is returned unchanged.
 * For 3 channels uses BGR2GRAY; for 4 channels uses BGRA2GRAY.
 *
 * @return Callable returning `CV_8U` or same depth with 1 channel.
 *
 * \ingroup type_and_channel_adapters
 */
inline auto to_gray() {
  return [](const cv::Mat& in) {
    if (in.channels() == 1) return in;
    cv::Mat out;
    int code = (in.channels() == 3) ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY;
    cv::cvtColor(in, out, code);
    return out;
  };
}

/**
 * @brief Promote a real image to a 2-channel complex image `[real, imag]`.
 *
 * The imaginary channel is initialized to zeros and matches input depth/type.
 *
 * @return Callable producing a 2-channel complex image.
 *
 * \ingroup type_and_channel_adapters
 */
inline auto to_complex() {
  return [](const cv::Mat& in) {
    cv::Mat zeros = cv::Mat::zeros(in.size(), in.type());
    cv::Mat out;
    std::vector<cv::Mat> ch{in, zeros};
    cv::merge(ch, out);
    return out;
  };
}

/**
 * @brief Extract real part from a 2-channel complex image.
 *
 * @pre `in.channels() == 2`.
 * @return Callable returning the first (real) channel.
 *
 * \ingroup type_and_channel_adapters
 */
inline auto real() {
  return [](const cv::Mat& in) {
    CV_Assert(in.channels() == 2);
    std::vector<cv::Mat> ch(2);
    cv::split(in, ch);
    return ch[0];
  };
}

/**
 * @brief Extract imaginary part from a 2-channel complex image.
 *
 * @pre `in.channels() == 2`.
 * @return Callable returning the second (imaginary) channel.
 *
 * \ingroup type_and_channel_adapters
 */
inline auto imag() {
  return [](const cv::Mat& in) {
    CV_Assert(in.channels() == 2);
    std::vector<cv::Mat> ch(2);
    cv::split(in, ch);
    return ch[1];
  };
}

// ============================================================================
// 2) Scalar/Value Transforms (image–scalar or per-pixel value ops)
// ============================================================================

/**
 * @brief Multiply all pixels by a scalar.
 *
 * @param scale Multiplicative factor.
 * @return Callable performing `in * scale`.
 *
 * \ingroup scalar_value_transforms
 */
inline auto multiply_scalar(double scale) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    in.convertTo(out, in.type(), scale);
    return out;
  };
}

/**
 * @brief Divide all pixels by a scalar.
 *
 * @param scale Divisor (must be non-zero).
 * @return Callable performing `in / scale`.
 *
 * \ingroup scalar_value_transforms
 */
inline auto divide_scalar(double scale) {
  CV_Assert(scale != 0.0);
  return [=](const cv::Mat& in) {
    cv::Mat out;
    in.convertTo(out, in.type(), 1.0 / scale);
    return out;
  };
}

/**
 * @brief Add a scalar value to all pixels.
 *
 * @param value Scalar to add (broadcast across channels).
 * @return Callable performing `in + value`.
 *
 * \ingroup scalar_value_transforms
 */
inline auto add_scalar(const cv::Scalar& value) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::add(in, value, out);
    return out;
  };
}

/**
 * @brief Subtract a scalar value from all pixels.
 *
 * @param value Scalar to subtract (broadcast across channels).
 * @return Callable performing `in - value`.
 *
 * \ingroup scalar_value_transforms
 */
inline auto subtract_scalar(const cv::Scalar& value) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::subtract(in, value, out);
    return out;
  };
}

/**
 * @brief Clamp pixel values to [minVal, maxVal], with optional type
 * preservation.
 *
 * For integer-depth inputs, temporary conversion to `CV_32F` is used unless
 * `preserve_type` is false; result is converted back to original type.
 *
 * @param minVal         Lower clamp bound (inclusive).
 * @param maxVal         Upper clamp bound (inclusive).
 * @param preserve_type  Preserve original depth via temporary float (default
 * true).
 * @return Callable performing clamping.
 *
 * \ingroup scalar_value_transforms
 */
inline auto clamp(double minVal = 0.0, double maxVal = 1.0,
                  bool preserve_type = true) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::Mat tmp;
    if (preserve_type &&
        (in.depth() == CV_8U || in.depth() == CV_16U || in.depth() == CV_16S))
      in.convertTo(tmp, CV_32F);
    else
      tmp = in;
    cv::min(cv::max(tmp, minVal), maxVal, out);
    if (preserve_type && tmp.type() != in.type()) out.convertTo(out, in.type());
    return out;
  };
}

/**
 * @brief Natural logarithm of (in + 1).
 *
 * Useful for simple log compression while avoiding log(0).
 *
 * @return Callable computing `log(in + 1)`.
 *
 * \ingroup scalar_value_transforms
 */
inline auto log() {
  return [](const cv::Mat& in) {
    cv::Mat out;
    cv::log(in + 1.0, out);
    return out;
  };
}

/**
 * @brief Normalize with OpenCV's `cv::normalize`.
 *
 * Default is min-max scaling to [a,b]. See OpenCV docs for `normType`/`dtype`.
 *
 * @param a        Lower bound (default 0.0).
 * @param b        Upper bound (default 1.0).
 * @param normType Normalization type (default `cv::NORM_MINMAX`).
 * @param dtype    Output type (default -1 = same type as input).
 * @return Callable performing normalization.
 *
 * \ingroup scalar_value_transforms
 */
inline auto normalize(double a = 0.0, double b = 1.0,
                      int normType = cv::NORM_MINMAX, int dtype = -1) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::normalize(in, out, a, b, normType, dtype);
    return out;
  };
}

/**
 * @brief Contrast stretch using histogram percentiles.
 *
 * Linearly maps intensities so that `lower_percent` of values are clipped to
 * black and `upper_percent` to white. Works on grayscale or color images,
 * supporting 8U, 16U, and 32F depths.
 *
 * @param lower_percent  Lower tail cutoff (percentage)
 * @param upper_percent  Upper tail cutoff (percentage)
 * @param per_channel    Stretch each channel independently (else
 *                       luminance-based)
 * @return Callable performing contrast stretch.
 *
 * \ingroup scalar_value_transforms
 */
inline auto linear_contrast_stretch(float lower_percent = 2.0f,
                                    float upper_percent = 98.0f,
                                    bool per_channel = true) {
  return [=](const cv::Mat& in) {
    CV_Assert(in.channels() == 1 || in.channels() == 3);
    CV_Assert(in.depth() == CV_8U || in.depth() == CV_16U ||
              in.depth() == CV_32F);

    const int bins = (in.depth() == CV_8U)    ? 256
                     : (in.depth() == CV_16U) ? 65536
                                              : 65536;
    const float range[] = {0.0f, static_cast<float>(bins)};
    const float* histRange = {range};

    auto stretch_channel = [&](const cv::Mat& ch) {
      cv::Mat hist;
      cv::calcHist(&ch, 1, 0, cv::Mat(), hist, 1, &bins, &histRange, true,
                   false);
      std::vector<float> cdf(bins);
      cdf[0] = hist.at<float>(0);
      for (int i = 1; i < bins; ++i) cdf[i] = cdf[i - 1] + hist.at<float>(i);
      float total = cdf.back();
      float lo_thresh = lower_percent / 100.0f * total;
      float hi_thresh = upper_percent / 100.0f * total;

      int lo = 0, hi = bins - 1;
      for (int i = 0; i < bins; ++i)
        if (cdf[i] >= lo_thresh) {
          lo = i;
          break;
        }
      for (int i = bins - 1; i >= 0; --i)
        if (cdf[i] <= hi_thresh) {
          hi = i;
          break;
        }

      cv::Mat out;
      if (in.depth() == CV_32F) {
        out = (ch - float(lo)) / float(hi - lo);
        cv::min(cv::max(out, 0.0f), 1.0f, out);
      } else {
        double scale = (hi == lo) ? 1.0 : double(bins - 1) / (hi - lo);
        ch.convertTo(out, ch.type(), scale, -lo * scale);
        cv::threshold(out, out, bins - 1, bins - 1, cv::THRESH_TRUNC);
        cv::threshold(out, out, 0, 0, cv::THRESH_TOZERO);
      }
      return out;
    };

    if (in.channels() == 1) return stretch_channel(in);

    std::vector<cv::Mat> ch;
    cv::split(in, ch);

    if (per_channel) {
      for (auto& c : ch) c = stretch_channel(c);
    } else {
      // Luminance approach
      cv::Mat lum;
      cv::cvtColor(in, lum, cv::COLOR_BGR2GRAY);
      cv::Mat stretch_lum = stretch_channel(lum);

      std::vector<cv::Mat> ch_f(3);
      for (int i = 0; i < 3; ++i) ch[i].convertTo(ch_f[i], CV_32F);
      cv::Mat lum_f, stretch_f;
      lum.convertTo(lum_f, CV_32F);
      stretch_lum.convertTo(stretch_f, CV_32F);
      cv::Mat ratio = stretch_f / (lum_f + 1e-6f);

      for (int i = 0; i < 3; ++i) ch_f[i] = ch_f[i].mul(ratio);

      for (int i = 0; i < 3; ++i) ch_f[i].convertTo(ch[i], in.depth());
    }

    cv::Mat out;
    cv::merge(ch, out);
    return out;
  };
}

/**
 * @brief Histogram equalization for contrast enhancement.
 *
 * Works on grayscale or color images (3-channel). For color inputs, performs
 * per-channel equalization or luminance-based equalization in YCrCb space.
 *
 * Supports CV_8U, CV_16U, and CV_32F inputs. Output retains same type/depth.
 * For float data, values are assumed in [0, 1] and scaled to 8-bit internally.
 *
 * @param per_channel  Apply equalization independently to each channel
 *                     (default is true).
 * @return Callable that performs histogram equalization.
 *
 * \ingroup scalar_value_transforms
 */
inline auto histogram_equalize_stretch(bool per_channel = true) {
  return [=](const cv::Mat& in) {
    CV_Assert(in.channels() == 1 || in.channels() == 3);
    CV_Assert(in.depth() == CV_8U || in.depth() == CV_16U ||
              in.depth() == CV_32F);

    auto equalize_single_channel = [](const cv::Mat& ch) -> cv::Mat {
      cv::Mat ch8u, eq;
      if (ch.depth() == CV_8U)
        ch8u = ch;
      else if (ch.depth() == CV_16U)
        ch.convertTo(ch8u, CV_8U, 1.0 / 256.0);  // scale 16U → 8U
      else if (ch.depth() == CV_32F)
        ch.convertTo(ch8u, CV_8U, 255.0);  // assume [0,1] float

      cv::equalizeHist(ch8u, eq);

      // Convert back to original type
      cv::Mat out;
      if (ch.depth() == CV_8U)
        out = eq;
      else if (ch.depth() == CV_16U)
        eq.convertTo(out, CV_16U, 256.0);  // scale 8U → 16U
      else if (ch.depth() == CV_32F)
        eq.convertTo(out, CV_32F, 1.0 / 255.0);  // scale 8U → float
      return out;
    };

    // Grayscale
    if (in.channels() == 1) return equalize_single_channel(in);

    // Color (3-channel)
    std::vector<cv::Mat> ch;
    cv::split(in, ch);

    if (per_channel) {
      for (auto& c : ch) c = equalize_single_channel(c);
      cv::Mat out;
      cv::merge(ch, out);
      return out;
    } else {
      // Luminance equalization in YCrCb space
      cv::Mat ycrcb, eq_ycrcb;
      cv::cvtColor(in, ycrcb, cv::COLOR_BGR2YCrCb);
      std::vector<cv::Mat> ych;
      cv::split(ycrcb, ych);
      ych[0] = equalize_single_channel(ych[0]);  // equalize Y only
      cv::merge(ych, eq_ycrcb);
      cv::Mat out;
      cv::cvtColor(eq_ycrcb, out, cv::COLOR_YCrCb2BGR);
      return out;
    }
  };
}

// ============================================================================
// 3) Element-wise Arithmetic Operations
// ============================================================================

/**
 * @brief Weighted sum: `alpha * in + beta * other + gamma`.
 *
 * Uses `cv::addWeighted` on the first element of each Scalar.
 *
 * @param other  Second image (same size and type).
 * @param alpha  Weight for `in` (default 1.0).
 * @param beta   Weight for `other` (default 1.0).
 * @param gamma  Scalar bias term (default 0.0).
 * @return Callable performing the weighted sum.
 *
 * \ingroup elementwise_operations
 */
inline auto add(const cv::Mat& other, const cv::Scalar& alpha = cv::Scalar(1.0),
                const cv::Scalar& beta = cv::Scalar(1.0),
                const cv::Scalar& gamma = cv::Scalar(0.0)) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::addWeighted(in, alpha[0], other, beta[0], gamma[0], out);
    return out;
  };
}

/**
 * @brief Element-wise subtraction: `in - other`, with optional scaling.
 *
 * @param other  Image to subtract.
 * @param scale  Optional multiplicative scale applied after subtraction.
 * @return Callable performing the subtraction.
 *
 * \ingroup elementwise_operations
 */
inline auto subtract(const cv::Mat& other, double scale = 1.0) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::subtract(in, other, out, cv::noArray(), in.type());
    if (scale != 1.0) out *= scale;
    return out;
  };
}

/**
 * @brief Element-wise multiplication with optional scale.
 *
 * @param other Image to multiply with.
 * @param scale Optional multiplicative factor (default 1.0).
 * @return Callable performing the multiplication `in .* other * scale`.
 *
 * \ingroup elementwise_operations
 */
inline auto multiply(const cv::Mat& other, double scale = 1.0) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::multiply(in, other, out, scale);
    return out;
  };
}

/**
 * @brief Element-wise division with optional scale.
 *
 * @param other Denominator image.
 * @param scale Optional multiplicative factor (default 1.0).
 * @return Callable performing the division `in ./ other * scale`.
 *
 * \ingroup elementwise_operations
 */
inline auto divide(const cv::Mat& other, double scale = 1.0) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::divide(in, other, out, scale);
    return out;
  };
}

// ============================================================================
// 4) Complex-Domain Operations
// ============================================================================

/**
 * @brief Complex conjugate of a 2-channel image.
 *
 * @pre `in.channels() == 2`.
 * @return Callable negating the imaginary channel.
 *
 * \ingroup complex_domain_operations
 */
inline auto conjugate() {
  return [](const cv::Mat& in) {
    CV_Assert(in.channels() == 2);
    std::vector<cv::Mat> ch(2);
    cv::split(in, ch);
    ch[1] = -ch[1];
    cv::Mat out;
    cv::merge(ch, out);
    return out;
  };
}

/**
 * @brief Magnitude of a 2-channel complex image: `sqrt(real^2 + imag^2)`.
 *
 * @return Callable producing single-channel magnitude.
 *
 * @pre `in.channels() == 2`.
 *
 * \ingroup complex_domain_operations
 */
inline auto magnitude() {
  return [](const cv::Mat& in) {
    CV_Assert(in.channels() == 2);
    std::vector<cv::Mat> p(2);
    cv::split(in, p);
    cv::Mat out;
    cv::magnitude(p[0], p[1], out);
    return out;
  };
}

/**
 * @brief Phase/angle of a 2-channel complex image in (-π, π].
 *
 * Uses `cv::phase` then wraps to the symmetric interval by subtracting one full
 * period where needed.
 *
 * @param degrees If true, output is in degrees; otherwise radians.
 * @return Callable producing phase image.
 *
 * @pre `in.channels() == 2`.
 *
 * \ingroup complex_domain_operations
 */
inline auto phase(bool degrees = false) {
  return [=](const cv::Mat& in) {
    CV_Assert(in.channels() == 2);
    std::vector<cv::Mat> p(2);
    cv::split(in, p);
    cv::Mat out;
    cv::phase(p[0], p[1], out, degrees);
    const double half = degrees ? 180.0 : CV_PI;
    const double period = degrees ? 360.0 : 2.0 * CV_PI;
    cv::Mat mask = (out >= half);
    cv::subtract(out, period, out, mask);
    return out;
  };
}

/**
 * @brief Power spectrum `|F|^2` for complex images, optionally log scaled.
 *
 * @param log_scale If true, returns `log(|F|^2 + epsilon)`.
 * @param epsilon   Stabilizer inside the log (default 1e-12).
 * @return Callable producing single-channel power spectrum image.
 *
 * @pre `in.channels() == 2`.
 *
 * \ingroup complex_domain_operations
 */
inline auto power_spectrum(bool log_scale = true, double epsilon = 1e-12) {
  // Power spectrum: |F|^2 (log optional)
  return [=](const cv::Mat& in) {
    CV_Assert(in.channels() == 2);
    std::vector<cv::Mat> ch(2);
    cv::split(in, ch);
    cv::Mat out;
    cv::multiply(ch[0], ch[0], out);
    cv::Mat tmp;
    cv::multiply(ch[1], ch[1], tmp);
    out += tmp;
    if (log_scale) {
      cv::log(out + epsilon, out);
    }
    return out;
  };
}

/**
 * @brief Complex multiplication of two 2-channel images.
 *
 * Computes `(a+ib) * (c+id)` channel-wise.
 *
 * @param other Complex image with 2 channels.
 * @return Callable producing complex product.
 *
 * @pre Both inputs must have 2 channels.
 *
 * \ingroup complex_domain_operations
 */
inline auto complex_multiply(const cv::Mat& other) {
  return [=](const cv::Mat& in) {
    CV_Assert(in.channels() == 2 && other.channels() == 2);
    std::vector<cv::Mat> a(2), b(2), result(2);
    cv::split(in, a);
    cv::split(other, b);
    result[0] = a[0].mul(b[0]) - a[1].mul(b[1]);  // real
    result[1] = a[0].mul(b[1]) + a[1].mul(b[0]);  // imag
    cv::Mat out;
    cv::merge(result, out);
    return out;
  };
}

/**
 * @brief Complex division of two 2-channel images with epsilon stabilization.
 *
 * Computes `(a+ib) / (c+id)` using `(ac+bd)/(c^2+d^2+epsilon)` for the real
 * part and `(bc-ad)/(c^2+d^2+epsilon)` for the imaginary part of the quotient.
 *
 * @param other   Complex denominator with 2 channels.
 * @param epsilon Small value added to denominator to avoid division by zero.
 * @return Callable producing complex quotient.
 *
 * @pre Both inputs must have 2 channels.
 *
 * \ingroup complex_domain_operations
 */
inline auto complex_divide(const cv::Mat& other, double epsilon = 1e-12) {
  return [=](const cv::Mat& in) {
    CV_Assert(in.channels() == 2 && other.channels() == 2);
    std::vector<cv::Mat> a(2), b(2), result(2);
    cv::split(in, a);
    cv::split(other, b);
    cv::Mat denominator = b[0].mul(b[0]) + b[1].mul(b[1]) + epsilon;
    result[0] = (a[0].mul(b[0]) + a[1].mul(b[1])) / denominator;  // real
    result[1] = (a[1].mul(b[0]) - a[0].mul(b[1])) / denominator;  // imag
    cv::Mat out;
    cv::merge(result, out);
    return out;
  };
}

// ============================================================================
// 5) Fourier/Frequency-Domain Utilities
// ============================================================================

/**
 * @brief Pad image to even dimensions using a constant (zero) border.
 *
 * If both dimensions are already even, the input is returned unchanged.
 *
 * @return Callable producing an even-sized image via right/bottom padding.
 *
 * \ingroup fourier_frequency_domain_utilities
 */
inline auto pad_to_even() {
  return [](const cv::Mat& in) {
    int newRows = (in.rows % 2 == 0) ? in.rows : in.rows + 1;
    int newCols = (in.cols % 2 == 0) ? in.cols : in.cols + 1;
    if (newRows == in.rows && newCols == in.cols) return in;
    cv::Mat out;
    cv::copyMakeBorder(in, out, 0, newRows - in.rows, 0, newCols - in.cols,
                       cv::BORDER_CONSTANT, cv::Scalar(0));
    return out;
  };
}

/**
 * @brief Discrete Fourier Transform producing complex output.
 *
 * @param flags OpenCV DFT flags (e.g., `cv::DFT_SCALE`, `cv::DFT_ROWS`).
 *              `cv::DFT_COMPLEX_OUTPUT` is always enforced.
 * @return Callable performing DFT.
 *
 * \ingroup fourier_frequency_domain_utilities
 */
inline auto dft(int flags = 0) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::dft(in, out, flags | cv::DFT_COMPLEX_OUTPUT);
    return out;
  };
}

/**
 * @brief Inverse Discrete Fourier Transform producing real output.
 *
 * The default flag includes `cv::DFT_SCALE` to scale the result by `1/N` (for
 * orthonormality).
 *
 * @param flags Defaults to `cv::DFT_SCALE`. Note that `cv::DFT_INVERSE`
 *              and `cv::DFT_REAL_OUTPUT` are always applied.
 * @return Callable performing the inverse DFT.
 *
 * \ingroup fourier_frequency_domain_utilities
 */
inline auto idft(int flags = cv::DFT_SCALE) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::dft(in, out, flags | cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    return out;
  };
}

/**
 * @brief Swap DFT quadrants so that the zero-frequency component moves to
 * the center.
 *
 * Works on 2-dimensional cv::Mats (even dimensions recommended).
 *
 * @return Callable performing quadrant swaps (Q0<->Q3 and Q1<->Q2).
 *
 * \ingroup fourier_frequency_domain_utilities
 */
inline auto fft_shift() {
  return [](const cv::Mat& in) {
    cv::Mat out = in.clone();
    int cx = out.cols / 2;
    int cy = out.rows / 2;
    cv::Mat q0(out, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(out, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(out, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(out, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    return out;
  };
}

// ============================================================================
// 6) Spatial Utilities
// ============================================================================

/**
 * @brief Circularly shift (roll) rows and columns.
 *
 * Positive shifts move content down/right; negative move content up/left.
 *
 * @param shift_rows Row shift (can be negative).
 * @param shift_cols Column shift (can be negative).
 * @return Callable producing a rolled/shifted image.
 *
 * @pre `in.dims == 2`.
 *
 * \ingroup spatial_utilities
 */
inline auto roll(int shift_rows, int shift_cols) {
  return [=](const cv::Mat& in) {
    CV_Assert(in.dims == 2);

    const int rows = in.rows;
    const int cols = in.cols;

    int dy = ((shift_rows % rows) + rows) % rows;
    int dx = ((shift_cols % cols) + cols) % cols;

    if (dy == 0 && dx == 0) return in;

    cv::Mat out(in.size(), in.type());

    cv::Mat src1 = in(cv::Rect(0, 0, cols - dx, rows - dy));
    cv::Mat src2 = in(cv::Rect(cols - dx, 0, dx, rows - dy));
    cv::Mat src3 = in(cv::Rect(0, rows - dy, cols - dx, dy));
    cv::Mat src4 = in(cv::Rect(cols - dx, rows - dy, dx, dy));

    src1.copyTo(out(cv::Rect(dx, dy, cols - dx, rows - dy)));
    src2.copyTo(out(cv::Rect(0, dy, dx, rows - dy)));
    src3.copyTo(out(cv::Rect(dx, 0, cols - dx, dy)));
    src4.copyTo(out(cv::Rect(0, 0, dx, dy)));

    return out;
  };
}

/**
 * @brief Affine warp defined by rotation (deg), scale, and translations.
 *
 * Builds a 2×3 matrix via `cv::getRotationMatrix2D(center, angle_degrees,
 * scale)` about the input image center, then adds translations (tx, ty)
 * in pixels.
 *
 * @param angle_degrees   Rotation angle in degrees (CCW).
 * @param scale           Uniform scale factor (1.0 = no scaling).
 * @param tx              Horizontal translation in pixels (right positive).
 * @param ty              Vertical translation in pixels (down positive).
 * @param dsize           Output size; (0,0) => same as input.
 * @param flags           Interpolation/warp flags (e.g. cv::INTER_LINEAR).
 * @param borderMode      Border handling (e.g. cv::BORDER_CONSTANT).
 * @param borderValue     Border value used with CONSTANT border mode.
 * @return Callable applying the specified affine transform.
 *
 * \ingroup spatial_utilities
 */
inline auto warp_affine(double angle_degrees, double scale, double tx,
                        double ty, cv::Size dsize = cv::Size(),
                        int flags = cv::INTER_LINEAR,
                        int borderMode = cv::BORDER_CONSTANT,
                        const cv::Scalar& borderValue = cv::Scalar()) {
  return [=](const cv::Mat& in) {
    const cv::Size outSize =
        (dsize.width > 0 && dsize.height > 0) ? dsize : in.size();
    // Build rotation/scale about image center
    const cv::Point2f center(static_cast<float>(in.cols) * 0.5f,
                             static_cast<float>(in.rows) * 0.5f);
    cv::Mat M = cv::getRotationMatrix2D(center, angle_degrees, scale);
    // Add translations (pixels)
    M.at<double>(0, 2) += tx;
    M.at<double>(1, 2) += ty;
    cv::Mat out;
    cv::warpAffine(in, out, M, outSize, flags, borderMode, borderValue);
    return out;
  };
}

/**
 * @brief Polar (or log-polar) warp from Cartesian input using cv::warpPolar.
 *
 * This maps a Cartesian image to its polar representation. The x dimension
 * of the output parameterizes angle in [0, 2π), and the y dimension
 * parameterizes radius in [0, maxRadius].
 *
 * @param log_polar   If true, use logarithmic radius mapping; otherwise linear.
 * @param center      Polar center; if either component is negative, uses
 *                    image center.
 * @param maxRadius   Maximum radius in pixels; if <= 0, uses farthest corner.
 * @param dsize       Output size; if (0,0), uses input size (same rows/cols).
 * @param flags       Interpolation flags (e.g., cv::INTER_LINEAR).
 * @return Callable producing the polar/log-polar image.
 *
 * \ingroup spatial_utilities
 */
inline auto warp_polar(bool log_polar = false,
                       cv::Point2f center = {-1.f, -1.f},
                       double maxRadius = -1.0, cv::Size dsize = cv::Size(),
                       int flags = cv::INTER_LINEAR) {
  return [=](const cv::Mat& in) {
    // Resolve output sampling (Angle x Radius)
    const cv::Size outSize =
        (dsize.width > 0 && dsize.height > 0) ? dsize : in.size();
    // Resolve center (Default: image center)
    cv::Point2f c = center;
    if (c.x < 0.f || c.y < 0.f) {
      c = cv::Point2f(static_cast<float>(in.cols) * 0.5f,
                      static_cast<float>(in.rows) * 0.5f);
    }
    // Resolve max radius (default: farthest corner to cover whole image)
    double mr = maxRadius;
    if (mr <= 0.0) {
      const cv::Point2f tl(0.f, 0.f), tr(static_cast<float>(in.cols), 0.f);
      const cv::Point2f bl(0.f, static_cast<float>(in.rows));
      const cv::Point2f br(static_cast<float>(in.cols),
                           static_cast<float>(in.rows));
      auto dist = [&](const cv::Point2f& p) { return cv::norm(p - c); };
      mr = std::max({dist(tl), dist(tr), dist(bl), dist(br)});
    }
    // Mode: linear or logarithmic radial mapping; no inverse mapping exposed
    const int mode = (log_polar ? cv::WARP_POLAR_LOG : cv::WARP_POLAR_LINEAR);
    cv::Mat out;
    out = cv::Mat::zeros(outSize, in.type());
    cv::warpPolar(in, out, outSize, c, mr, flags | mode);
    return out;
  };
}

/**
 * @brief Warp a quadrilateral region to a new quadrilateral using
 *        4-point correspondence.
 *
 * Computes the 3×3 homography from source→destination points and applies
 * the perspective warp. Works with 8U/16U/32F images and preserves type.
 *
 * @param src_pts       Four source `Point2f` coordinates.
 * @param dst_pts       Four destination `Point2f` coordinates.
 * @param dsize         Output image size (default: same as input).
 * @param flags         Interpolation flags (default: `cv::INTER_LINEAR`).
 * @param borderMode    Border behavior (default: `cv::BORDER_CONSTANT`).
 * @param borderValue   Fill value for out-of-bounds (default: 0).
 * @return Callable that performs the full perspective warp.
 *
 * \ingroup spatial_utilities
 */
inline auto warp_quad(const std::vector<cv::Point2f>& src_pts,
                      const std::vector<cv::Point2f>& dst_pts,
                      cv::Size dsize = {}, int flags = cv::INTER_LINEAR,
                      int borderMode = cv::BORDER_CONSTANT,
                      const cv::Scalar& borderValue = cv::Scalar()) {
  CV_Assert(src_pts.size() == 4 && dst_pts.size() == 4);
  return [=](const cv::Mat& in) {
    cv::Mat H = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Size target = (dsize.width > 0 && dsize.height > 0) ? dsize : in.size();
    cv::Mat out;
    cv::warpPerspective(in, out, H, target, flags, borderMode, borderValue);
    return out;
  };
}

/**
 * @brief Apply a Gaussian blur (low-pass filter) to the image.
 *
 * Uses OpenCV's `cv::GaussianBlur` with the specified kernel size and standard
 * deviations. This is useful for noise reduction, anti-aliasing, or preparing
 * an image for edge detection.
 *
 * The function supports anisotropic filtering (different values for sigmaX and
 * sigmaY), and if `sigmaY == 0`, OpenCV uses the same value as `sigmaX`.
 *
 * @param ksize   Size of Gaussian kernel (must be odd and positive in each
 *                dimension).
 * @param sigmaX  Gaussian kernel standard deviation in the X direction.
 * @param sigmaY  Gaussian kernel standard deviation in the Y direction
 *                (default is 0 which indicated this should be the same as
 *                sigmaX).
 * @return Callable that applies Gaussian blur.
 *
 * \ingroup spatial_utilities
 */
inline auto gaussian_blur(cv::Size ksize, double sigmaX, double sigmaY = 0) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::GaussianBlur(in, out, ksize, sigmaX, sigmaY);
    return out;
  };
}

/**
 * @brief Apply the Sobel operator to compute image gradients.
 *
 * Computes the first, second, or mixed derivatives of an image using
 * `cv::Sobel`. Useful for detecting edges or computing gradient magnitude
 * and direction.
 *
 * The Sobel operator is a discrete differentiation kernel, combining Gaussian
 * smoothing and differentiation.
 *
 * @param dx        Order of the derivative in the x-direction (e.g. 1 for
 *                  ∂/∂x).
 * @param dy        Order of the derivative in the y-direction (e.g. 0 for
 *                  ∂/∂x only).
 * @param ksize     Size of the extended Sobel kernel (must be 1, 3, 5, or 7).
 * @param scale     Optional scaling factor applied to the derivative
 *                  (default 1.0).
 * @param delta     Optional bias added to the result (default is 0.0).
 * @param ddepth    Desired output depth (e.g. `CV_16S`, `CV_32F`,
 *                  or -1 to match input).
 * @return Callable that applies the Sobel operator.
 *
 * \ingroup spatial_utilities
 */
inline auto sobel(int dx, int dy, int ksize = 3, double scale = 1.0,
                  double delta = 0.0, int ddepth = -1) {
  return [=](const cv::Mat& in) {
    cv::Mat out;
    cv::Sobel(in, out, ddepth, dx, dy, ksize, scale, delta);
    return out;
  };
}

}  // namespace pcv

#endif  // PIPELINE_CV_H_
