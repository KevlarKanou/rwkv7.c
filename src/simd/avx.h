#ifndef __AVX_H__
#define __AVX_H__
#include <immintrin.h>

static float _avx_horizontal_sum(__m256 v) {
    __m128 v1 = _mm256_extractf128_ps(v, 0);
    __m128 v2 = _mm256_extractf128_ps(v, 1);
    v1 = _mm_add_ps(v1, v2);
    v1 = _mm_hadd_ps(v1, v1);
    v1 = _mm_hadd_ps(v1, v1);
    return _mm_cvtss_f32(v1);
}

void vec_add(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 add_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(xout + i, add_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] + b[i]; }
}

void vec_sub(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 sub_vec = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(xout + i, sub_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] - b[i]; }
}

void vec_hadamard(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 prod_vec = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(xout + i, prod_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] * b[i]; }
}

void vec_bias(float *xout, const float *a, float b, int len) {
    __m256 b_vec = _mm256_broadcast_ss(&b);
    int i;
    for (i = 0; i <= len - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 add_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(xout + i, add_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] + b; }
}

void vec_scale(float *xout, const float *a, float b, int len) {
    __m256 b_vec = _mm256_broadcast_ss(&b);
    int i;
    for (i = 0; i <= len - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 prod_vec = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(xout + i, prod_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] * b; }
}

static float vec_dot_product(const float *a, const float *b, int len) {
    float ret = 0.0;
    int i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    for (; i <= len - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    ret = _avx_horizontal_sum(sum_vec);
    for (; i < len; i++) { ret += a[i] * b[i]; }
    return ret;
}

void vec_out_product(float *xout, const float *a, const float *b, int len) {
    for (int i = 0; i < len; i++) {
        int j = 0;
        const __m256 a_vec = _mm256_set1_ps(a[i]);
        for (; j <= len - 8; j += 8) {
            const __m256 b_vec = _mm256_loadu_ps(b + j);
            const __m256 result = _mm256_mul_ps(a_vec, b_vec);
            _mm256_storeu_ps(&xout[i * len + j], result);
        }
        for (; j < len; j++) { xout[i * len + j] = a[i] * b[j]; }
    }
}

float vec_sum(const float *x, int len) {
    float ret = 0.0f;
    int i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    for (; i <= len - 8; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        sum_vec = _mm256_add_ps(sum_vec, x_vec);
    }
    ret = _avx_horizontal_sum(sum_vec);
    for (; i < len; i++) { ret += x[i]; }
    return ret;
}

void lerp(float *xout, const float *x, const float *last_x, const float *mu, int len) {
    int i = 0;
    for (; i <= len - 8; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 last_x_vec = _mm256_loadu_ps(last_x + i);
        __m256 mu_vec = _mm256_loadu_ps(mu + i);

        __m256 xout_vec = _mm256_sub_ps(last_x_vec, x_vec);
        xout_vec = _mm256_fmadd_ps(mu_vec, xout_vec, x_vec);
        _mm256_storeu_ps(xout + i, xout_vec);
    }
    for (; i < len; i++) {
        xout[i] = x[i] + mu[i] * (last_x[i] - x[i]);
    }
}

#endif
