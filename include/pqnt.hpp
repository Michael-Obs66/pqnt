//
// pqnt.h

#ifndef PQNT_HPP
#define PQNT_HPP

#include <vector>
#include <cstdint>
#include <cstddef>   // <-- for ssize_t

#ifndef ssize_t
typedef long ssize_t;
#endif

class pqnt {
public:

    // Power Transform: t = sign(x) * |x|^p
    static void power_transform(
        const std::vector<float> &x,
        std::vector<float> &t,
        float p
    );

    // Scale: max(|t|)/127
    static float compute_scale(
        const std::vector<float> &t
    );

    // Quantize to int8
    static void quantize(
        const std::vector<float> &t,
        std::vector<int8_t> &q,
        float s
    );

    // Dequantize int8 back to float
    static void dequantize(
        const std::vector<int8_t> &q,
        std::vector<float> &t_tilde,
        float s
    );

    // Inverse transform: x_hat = sign(t)*|t|^(1/p)
    static void inverse_transform(
        const std::vector<float> &t_tilde,
        std::vector<float> &xhat,
        float p
    );

    // Full pipeline: returns MAE, produces xhat and scale
    static float run(
        const std::vector<float> &x,
        float p,
        std::vector<float> &xhat,
        float &s_out
    );

    // Per-channel quantization (ND-tensor, any axis)
    static void quantize_per_channel(
        const std::vector<float> &x,
        std::vector<float> &out,
        const std::vector<ssize_t> &shape,   // <-- FIXED (must match pybind11)
        int axis,
        float p
    );
};


// Baseline quantization (no transform)
float baseline_quant(
    const std::vector<float> &x,
    std::vector<float> &xhat
);

#endif // PQNT_HPP

