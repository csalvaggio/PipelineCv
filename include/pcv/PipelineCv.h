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
#include <vector>

#include <opencv2/opencv.hpp>

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

}  // namespace pcv

#endif // PIPELINE_CV_H_
