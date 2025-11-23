// bindings.cpp 

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pqnt.hpp"

namespace py = pybind11;

// Helper: convert NumPy array -> std::vector<float>
static std::vector<float> ndarray_to_vec_float(const py::array &arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim < 1) {
        throw std::runtime_error("Input array must have at least 1 dimension");
    }
    size_t total = 1;
    for (auto s : buf.shape) {
        total *= (size_t)s;
    }

    std::vector<float> out(total);
    // asumsi input float32
    if (buf.format != py::format_descriptor<float>::format()) {
        throw std::runtime_error("Expected float32 array");
    }
    float *data = static_cast<float *>(buf.ptr);
    std::copy(data, data + total, out.begin());
    return out;
}

// Helper: allocate NumPy array with given shape & fill from std::vector<float>
static py::array vec_float_to_ndarray(const std::vector<float> &v,
                                      const std::vector<ssize_t> &shape) {
    py::array out(py::dtype::of<float>(), shape);
    py::buffer_info buf = out.request();
    float *data = static_cast<float *>(buf.ptr);
    size_t total = 1;
    for (auto s : shape) total *= (size_t)s;
    if (total != v.size()) {
        throw std::runtime_error("Shape size does not match vector size");
    }
    std::copy(v.begin(), v.end(), data);
    return out;
}

// Helper: allocate NumPy int8 array from std::vector<int8_t>
static py::array vec_int8_to_ndarray(const std::vector<int8_t> &v,
                                     const std::vector<ssize_t> &shape) {
    py::array out(py::dtype::of<int8_t>(), shape);
    py::buffer_info buf = out.request();
    int8_t *data = static_cast<int8_t *>(buf.ptr);
    size_t total = 1;
    for (auto s : shape) total *= (size_t)s;
    if (total != v.size()) {
        throw std::runtime_error("Shape size does not match vector size");
    }
    std::copy(v.begin(), v.end(), data);
    return out;
}

PYBIND11_MODULE(pqnt, m) {
    m.doc() = "Power-based quantization helpers (pquantizer)";

    // --------------------
    // power_transform(x, p)
    // --------------------
    m.def("power_transform",
          [](py::array x, float p) {
              py::buffer_info buf = x.request();
              std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
              auto vx = ndarray_to_vec_float(x);

              std::vector<float> t;
              pqnt::power_transform(vx, t, p);

              return vec_float_to_ndarray(t, shape);
          },
          py::arg("x"), py::arg("p"),
          "Apply signed power transform: sign(x) * |x|^p");

    // --------------------
    // inverse_transform(t_tilde, p)
    // --------------------
    m.def("inverse_transform",
          [](py::array t_tilde, float p) {
              py::buffer_info buf = t_tilde.request();
              std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
              auto vt = ndarray_to_vec_float(t_tilde);

              std::vector<float> xhat;
              pqnt::inverse_transform(vt, xhat, p);

              return vec_float_to_ndarray(xhat, shape);
          },
          py::arg("t_tilde"), py::arg("p"),
          "Inverse of power_transform in float domain");

    // --------------------
    // quantize_array(x, scale) -> int8 array
    // --------------------
    m.def("quantize_array",
          [](py::array x, float scale) {
              py::buffer_info buf = x.request();
              std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
              auto vx = ndarray_to_vec_float(x);

              std::vector<int8_t> q;
              pqnt::quantize(vx, q, scale);

              return vec_int8_to_ndarray(q, shape);
          },
          py::arg("x"), py::arg("scale"),
          "Quantize float array with given scale into int8");

    // --------------------
    // quantize_tensor(x, bitwidth) -> (q_int8, scale, zero_point)
    // --------------------
    m.def("quantize_tensor",
          [](py::array x, int bitwidth) {
              if (bitwidth != 8) {
                  throw std::runtime_error("Only 8-bit quantization supported in this binding");
              }

              py::buffer_info buf = x.request();
              std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
              auto vx = ndarray_to_vec_float(x);

              // compute scale from data (symmetric quant, -127..127)
              float s = pqnt::compute_scale(vx);

              std::vector<int8_t> q;
              pqnt::quantize(vx, q, s);

              py::array q_arr = vec_int8_to_ndarray(q, shape);
              int zero_point = 0;  // symmetric, no offset

              return py::make_tuple(q_arr, s, zero_point);
          },
          py::arg("x"), py::arg("bitwidth"),
          "Quantize tensor globally to int8 (symmetric). Returns (q, scale, zero_point)");

    // --------------------
    // quantize_tensor_per_channel(x, axis, p)
    //
    // NOTE:
    // - x: float32 numpy array (any shape)
    // - axis: channel dimension
    // - p: power exponent used inside per-channel pipeline (run)
    // Returns: reconstructed float32 (same shape) after
    // per-channel power->quant->dequant->inverse.
    // --------------------
    m.def("quantize_tensor_per_channel",
          [](py::array x, int axis, float p) {
              py::buffer_info buf = x.request();

              int ndim = (int)buf.ndim;
              if (ndim < 1) {
                  throw std::runtime_error("Input must have at least 1 dimension");
              }

              // normalisasi axis negatif
              if (axis < 0) axis += ndim;
              if (axis < 0 || axis >= ndim) {
                  throw std::runtime_error("Invalid axis");
              }

              std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
              auto vx = ndarray_to_vec_float(x);
              std::vector<float> out;

              pqnt::quantize_per_channel(vx, out, shape, axis, p);

              return vec_float_to_ndarray(out, shape);
          },
          py::arg("x"), py::arg("axis"), py::arg("p"),
          "Per-channel pqnt (power+quant+dequant+inverse) along given axis");

    // --------------------
    // run(x, p) -> (xhat, scale, mae)
    // 1D or ND: flattened internally
    // --------------------
    m.def("run",
          [](py::array x, float p) {
              py::buffer_info buf = x.request();
              std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
              auto vx = ndarray_to_vec_float(x);

              std::vector<float> xhat;
              float s_out = 0.0f;
              float mae = pqnt::run(vx, p, xhat, s_out);

              py::array xhat_arr = vec_float_to_ndarray(xhat, shape);

              return py::make_tuple(xhat_arr, s_out, mae);
          },
          py::arg("x"), py::arg("p"),
          "Full 1D/ND pipeline: power->scale->quant->dequant->inverse, "
          "returns (reconstructed_x, scale, MAE)");

    // --------------------
    // baseline_quant(x) -> (xhat, mae)
    // --------------------
    m.def("baseline_quant",
          [](py::array x) {
              py::buffer_info buf = x.request();
              std::vector<ssize_t> shape(buf.shape.begin(), buf.shape.end());
              auto vx = ndarray_to_vec_float(x);

              std::vector<float> xhat;
              float mae = baseline_quant(vx, xhat);

              py::array xhat_arr = vec_float_to_ndarray(xhat, shape);
              return py::make_tuple(xhat_arr, mae);
          },
          py::arg("x"),
          "Baseline symmetric int8 quant (no power transform), returns (xhat, MAE)");
}
