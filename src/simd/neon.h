#ifndef __NEON_H__
#define __NEON_H__
#include <arm_neon.h>

static float _neon_horizontal_sum(float32x4_t v) {
    float32x2_t v1 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    v1 = vpadd_f32(v1, v1);
    return vget_lane_f32(v1, 0);
}

void vec_add(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 4; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t add_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(xout + i, add_vec); 
    }
    for (; i < len; i++) { xout[i] = a[i] + b[i]; }
}

void vec_sub(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 4; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t sub_vec = vsubq_f32(a_vec, b_vec);
        vst1q_f32(xout + i, sub_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] - b[i]; }
}

void vec_hadamard(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 4; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t prod_vec = vmulq_f32(a_vec, b_vec);
        vst1q_f32(xout + i, prod_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] * b[i]; }
}

void vec_bias(float *xout, const float *a, float b, int len) {
    float32x4_t b_vec = vdupq_n_f32(b);
    int i;
    for (i = 0; i <= len - 4; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t add_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(xout + i, add_vec); 
    }
    for (; i < len; i++) { xout[i] = a[i] + b; }
}

void vec_scale(float *xout, const float *a, float b, int len) {
    float32x4_t b_vec = vdupq_n_f32(b);
    int i;
    for (i = 0; i <= len - 4; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t prod_vec = vmulq_f32(a_vec, b_vec);
        vst1q_f32(xout + i, prod_vec); 
    }
    for (; i < len; i++) { xout[i] = a[i] * b; }
}

static float vec_dot_product(const float *a, const float *b, int len) {
    float ret = 0.0;
    int i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0);
    for (; i <= len - 4; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        sum_vec = vfmaq_f32(sum_vec, w_vec, x_vec);
    }
    ret = _neon_horizontal_sum(sum_vec);
    for (; i < len; i++) { ret += a[i] * b[i]; }
    return ret;
}

void vec_out_product(float *xout, const float *a, const float *b, int len) {
    for (int i = 0; i < len; i++) {
        int j = 0;
        const float32x4_t a_vec = vdupq_n_f32(a[i]);
        for (; j <= len - 4; j += 4) {
            const float32x4_t b_vec = vld1q_f32(b + j);
            const float32x4_t result = vmulq_f32(a_vec, b_vec);
            vst1q_f32(&xout[i * len + j], result);
        }
        for (; j < len; j++) { xout[i * len + j] = a[i] * b[j]; }
    }
}

float vec_sum(const float *x, int len) {
    float ret = 0.0f;
    int i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0);
    for (; i <= len - 4; i += 4) {
        float32x4_t x_vec = vld1q_f32(x + i);
        sum_vec = vaddq_f32(sum_vec, x_vec);
    }
    ret = _neon_horizontal_sum(sum_vec);
    for (; i < len; i++) { ret += x[i]; }
    return ret;
}

void lerp(float *xout, const float *x, const float *last_x, const float *mu, int len) {
    int i = 0;
    for (; i <= len - 4; i += 4) {
        float32x4_t x_vec = vld1q_f32(x + i);
        float32x4_t last_x_vec = vld1q_f32(last_x + i);
        float32x4_t mu_vec = vld1q_f32(mu + i);

        float32x4_t xout_vec = vsubq_f32(last_x_vec, x_vec);
        xout_vec = vfmaq_f32(x_vec, mu_vec, xout_vec);
        vst1q_f32(xout + i, xout_vec);
    }
    for (; i < len; i++) {
        xout[i] = x[i] + mu[i] * (last_x[i] - x[i]);
    }
}

#endif
