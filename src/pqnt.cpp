//
// pquant.cpp

#include "pqnt.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;

inline float sgn(float x) {
	return (x < 0.0f) ? -1.0f : 1.0f;
}


inline int8_t clip_int8(int v) {
	if (v > 127) return 127;
	if (v < -128) return -128;
	return (int8_t)v;
}


// implementation pquant

void pqnt::power_transform(
	const vector<float> &x,
	vector<float> &t,
	float p
) {
	size_t n = x.size();
	t.resize(n);
	for (size_t i = 0; i < n; i++) {
		float ax = fabsf(x[i]);
		if (ax == 0.0f) {
			t[i] = 0.0f;
		} else {
			float val = powf(ax, p);
			t[i] = (x[i] < 0 ? -val : val);
		}
	}
}

float pqnt::compute_scale(const vector<float> &t) {
	float maxabs = 0.0f;
	for (float v : t) maxabs = max(maxabs, fabsf(v));
	if (maxabs == 0.0f) return 1.0f;
	return maxabs / 127.0f;
}


void pqnt::quantize(
	const vector<float> &t,
	vector<int8_t> &q,
	float s
) {
	size_t n = t.size();
	q.resize(n);
	if (s <= 0.0f) s = 1.0f;
	for (size_t i=0; i < n; i++) {
		int r = (int)roundf(t[i] / s);
		q[i] = clip_int8(r);
	}
}

void pqnt::dequantize(
	const vector<int8_t> &q,
	vector<float> &t_tilde,
	float s
) {
	size_t n = q.size();
	t_tilde.resize(n);
	for (size_t i = 0; i < n; i++)
	    t_tilde[i] = (float)q[i] * s;
}

void pqnt::inverse_transform(
	const vector<float> &t_tilde,
	vector<float> &xhat,
	float p
) {
	size_t n = t_tilde.size();
	xhat.resize(n);
	float invp = 1.0f / p;

	for (size_t i = 0; i < n; i++) {
		float at = fabsf(t_tilde[i]);
		if (at == 0.0f) xhat[i] = 0.0f;
		else {
			float val = powf(at, invp);
			xhat[i] = (t_tilde[i] < 0 ? -val : val);
		}
	}
}


float pqnt::run(
	const vector<float> &x,
	float p,
	vector<float> &xhat,
	float &s_out
) {
	vector<float> t;
	vector<int8_t> q;
	vector<float> t_tilde;

	power_transform(x, t, p);
	float s = compute_scale(t);
	quantize(t, q, s);
	dequantize(q, t_tilde, s);
	inverse_transform(t_tilde, xhat, p);

	double acc = 0.0;
	for (size_t i = 0; i < x.size(); i++)
		acc += fabs((double)x[i] - (double)xhat[i]);

	s_out = s;
	return acc / (double)x.size();
}


// Baseline

float baseline_quant(
	const vector<float> &x,
	vector<float> &xhat
) {
	float maxabs = 0.0f;
	for (float v : x) maxabs = max(maxabs, fabsf(v));
	float s = (maxabs == 0 ? 1.0f : maxabs / 127.0f);

	xhat.resize(x.size());
	for (size_t i = 0; i < x.size(); i++) {
		int q = (int)roundf(x[i] / s);
		q = max(-128, min(127, q));
		xhat[i] = q* s;
	}

	double acc = 0.0;
	for (size_t i = 0; i < x.size(); i++)
		acc += fabs((double)x[i] - (double)xhat[i]);
	return acc / (double)x.size();
}	

//quantize_per_channel

void pqnt::quantize_per_channel(
    const vector<float> &x,
    vector<float> &out,
    const vector<ssize_t> &shape,
    int axis,
    float p
) {
    int ndim = shape.size();
    if (axis < 0 || axis >= ndim)
        throw runtime_error("Invalid axis");

    // Compute strides (C-order)
    vector<size_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * (size_t)shape[i + 1];

    size_t total = x.size();
    size_t channels = (size_t)shape[axis];

    out.resize(total);

    // -------- PER CHANNEL PROCESSING ----------
    for (size_t c = 0; c < channels; ++c) {

        vector<float> tmp;
        tmp.reserve(total / channels + 4);

        // GATHER
        for (size_t idx = 0; idx < total; ++idx) {
            size_t coord = (idx / strides[axis]) % (size_t)shape[axis];
            if (coord == c)
                tmp.push_back(x[idx]);
        }

        // QUANTIZE THIS CHANNEL (transform → scale → int8 → dequant → inverse)
        vector<float> qtmp;
        float scale_tmp;
        pqnt::run(tmp, p, qtmp, scale_tmp);

        // SCATTER BACK
        size_t k = 0;
        for (size_t idx = 0; idx < total; ++idx) {
            size_t coord = (idx / strides[axis]) % (size_t)shape[axis];
            if (coord == c)
                out[idx] = qtmp[k++];
        }
    }
}


































































	






	
	





















