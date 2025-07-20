#ifndef __NEON_FP16_H__
#define __NEON_FP16_H__
#include <arm_neon.h>

static float _neon_horizontal_sum_f32(float32x4_t v) {
    float32x2_t v1 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    v1 = vpadd_f32(v1, v1);
    return vget_lane_f32(v1, 0);
}

static float _neon_horizontal_sum_f16(float16x8_t v) {
    float16x4_t v1 = vadd_f16(vget_low_f16(v), vget_high_f16(v));
    v1 = vpadd_f16(v1, v1);
    v1 = vpadd_f16(v1, v1);
    return vget_lane_f16(v1, 0);
}

void vec_add(__fp16 *xout, const __fp16 *a, const __fp16 *b, int len) {
    int i = 0;
    for (; i <= len - 8; i += 8) {
        float16x8_t a_vec = vld1q_f16(a + i);
        float16x8_t b_vec = vld1q_f16(b + i);
        float16x8_t add_vec = vaddq_f16(a_vec, b_vec);
        vst1q_f16(xout + i, add_vec); 
    }
    for (; i < len; i++) { xout[i] = a[i] + b[i]; }
}

void vec_sub(__fp16 *xout, const __fp16 *a, const __fp16 *b, int len) {
    int i = 0;
    for (; i <= len - 8; i += 8) {
        float16x8_t a_vec = vld1q_f16(a + i);
        float16x8_t b_vec = vld1q_f16(b + i);
        float16x8_t sub_vec = vsubq_f16(a_vec, b_vec);
        vst1q_f16(xout + i, sub_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] - b[i]; }
}

void vec_hadamard(__fp16 *xout, const __fp16 *a, const __fp16 *b, int len) {
    int i = 0;
    for (; i <= len - 8; i += 8) {
        float16x8_t a_vec = vld1q_f16(a + i);
        float16x8_t b_vec = vld1q_f16(b + i);
        float16x8_t prod_vec = vmulq_f16(a_vec, b_vec);
        vst1q_f16(xout + i, prod_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] * b[i]; }
}

void vec_bias(__fp16 *xout, const __fp16 *a, __fp16 b, int len) {
    float16x8_t b_vec = vdupq_n_f16(b);
    int i = 0;
    for (; i <= len - 8; i += 8) {
        float16x8_t a_vec = vld1q_f16(a + i);
        float16x8_t add_vec = vaddq_f16(a_vec, b_vec);
        vst1q_f16(xout + i, add_vec); 
    }
    for (; i < len; i++) { xout[i] = a[i] + b; }
}

void vec_scale(__fp16 *xout, const __fp16 *a, __fp16 b, int len) {
    float16x8_t b_vec = vdupq_n_f16(b);
    int i = 0;
    for (; i <= len - 8; i += 8) {
        float16x8_t a_vec = vld1q_f16(a + i);
        float16x8_t prod_vec = vmulq_f16(a_vec, b_vec);
        vst1q_f16(xout + i, prod_vec); 
    }
    for (; i < len; i++) { xout[i] = a[i] * b; }
}

static float vec_dot_product(const __fp16 *a, const __fp16 *b, int len) {
    // pure fp16
    float ret = 0.0;
    int i = 0;
    float16x8_t sum_vec = vdupq_n_f16(0);
    for (; i <= len - 8; i += 8) {
        float16x8_t a_vec = vld1q_f16(a + i);
        float16x8_t b_vec = vld1q_f16(b + i);
        sum_vec = vfmaq_f16(sum_vec, a_vec, b_vec);
    }
    ret = _neon_horizontal_sum_f16(sum_vec);
    for (; i < len; i++) { ret += a[i] * b[i]; }
    return ret;

    // upgrade precision to fp32
    // float ret = 0.0;
    // int i = 0;
    // float32x4_t sum_vec_low = vdupq_n_f32(0);
    // float32x4_t sum_vec_high = vdupq_n_f32(0);
    
    // for (; i <= len - 8; i += 8) {
    //     float16x8_t a_vec = vld1q_f16(a + i);
    //     float16x8_t b_vec = vld1q_f16(b + i);
        
    //     float32x4_t a_low = vcvt_f32_f16(vget_low_f16(a_vec));
    //     float32x4_t a_high = vcvt_f32_f16(vget_high_f16(a_vec));
    //     float32x4_t b_low = vcvt_f32_f16(vget_low_f16(b_vec));
    //     float32x4_t b_high = vcvt_f32_f16(vget_high_f16(b_vec));
        
    //     sum_vec_low = vfmaq_f32(sum_vec_low, a_low, b_low);
    //     sum_vec_high = vfmaq_f32(sum_vec_high, a_high, b_high);
    // }
    
    // float32x4_t final_sum = vaddq_f32(sum_vec_low, sum_vec_high);
    // ret = _neon_horizontal_sum_f32(final_sum);
    
    // for (; i < len; i++) { 
    //     ret += (float)a[i] * (float)b[i]; 
    // }
    // return ret;
}

void vec_out_product(__fp16 *xout, const __fp16 *a, const __fp16 *b, int len) {
    for (int i = 0; i < len; i++) {
        int j = 0;
        const float16x8_t a_vec = vdupq_n_f16(a[i]);
        for (; j <= len - 8; j += 8) {
            const float16x8_t b_vec = vld1q_f16(b + j);
            const float16x8_t result = vmulq_f16(a_vec, b_vec);
            vst1q_f16(&xout[i * len + j], result);
        }
        for (; j < len; j++) { xout[i * len + j] = a[i] * b[j]; }
    }
}

float vec_sum(const __fp16 *x, int len) {
    float ret = 0.0f;
    int i = 0;
    float32x4_t sum_vec_low = vdupq_n_f32(0);
    float32x4_t sum_vec_high = vdupq_n_f32(0);
    for (; i <= len - 8; i += 8) {
        float16x8_t x_vec = vld1q_f16(x + i);
        float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x_vec));
        float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x_vec));
        sum_vec_low = vaddq_f32(sum_vec_low, x_low);
        sum_vec_high = vaddq_f32(sum_vec_high, x_high);
    }
    float32x4_t final_sum = vaddq_f32(sum_vec_low, sum_vec_high);
    ret = _neon_horizontal_sum_f32(final_sum);
    for (; i < len; i++) { ret += x[i]; }
    return ret;
}

void lerp(__fp16 *xout, const __fp16 *x, const __fp16 *last_x, const __fp16 *mu, int len) {
    int i = 0;
    for (; i <= len - 8; i += 8) {
        float16x8_t x_vec = vld1q_f16(x + i);
        float16x8_t last_x_vec = vld1q_f16(last_x + i);
        float16x8_t mu_vec = vld1q_f16(mu + i);

        float16x8_t xout_vec = vsubq_f16(last_x_vec, x_vec);
        xout_vec = vfmaq_f16(x_vec, mu_vec, xout_vec);
        vst1q_f16(xout + i, xout_vec);
    }
    for (; i < len; i++) {
        xout[i] = x[i] + mu[i] * (last_x[i] - x[i]);
    }
}

#endif
