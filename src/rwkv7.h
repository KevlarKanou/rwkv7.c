#ifndef __RWKV7_H__
#define __RWKV7_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "rwkv_vocab_v20230424.h"

#if defined(AVX)
#include "simd/avx.h"
#elif defined(NEON)
#include "simd/neon.h"
#endif

#define ERR(COND, MSG)              do { if (COND) { fprintf(stderr, "Error: %s\n", MSG); exit(EXIT_FAILURE); } } while(0)
#define SYSTIME_MS(X)               do { struct timespec time; clock_gettime(0, &time); X = time.tv_sec * 1000 + time.tv_nsec / 1000000; } while(0)
#define ARRLEN(X)                   (int)(sizeof(X)/sizeof(X[0]))
#define IDX(I, J, K, DIM2, DIM3)    ((I) * (DIM2) * (DIM3) + (J) * (DIM3) + (K))
#define SQUARE(X)                   ((X) * (X))
#define MAXIMUM(A, B)               ((A > B) ? A : B)
#define RELU(X)                     MAXIMUM(X, 0)
#define IS_NAN(X)                   (!((X) == (X)))

#define MATxVEC(XOUT, X, W)         mat_mul_vec(XOUT, X, W, ARRLEN(X), ARRLEN(XOUT))
#define VECADD(XOUT, A, B)          vec_add(XOUT, A, B, ARRLEN(XOUT))
#define VECSUB(XOUT, A, B)          vec_sub(XOUT, A, B, ARRLEN(XOUT))
#define HADAMARD(XOUT, A, B)        vec_hadamard(XOUT, A, B, ARRLEN(XOUT))
#define VECBIAS(XOUT, A, B)         vec_bias(XOUT, A, B, ARRLEN(XOUT))
#define VECSCALE(XOUT, A, B)        vec_scale(XOUT, A, B, ARRLEN(XOUT))
#define LERP(XOUT, X, LAST_X, MU)   lerp(XOUT, X, LAST_X, MU, ARRLEN(XOUT))

#define VECTANH(XOUT)               do { for (int i = 0; i < ARRLEN(XOUT); i++) { XOUT[i] = tanh(XOUT[i]); } } while(0)
#define VECSIGM(XOUT)               do { for (int i = 0; i < ARRLEN(XOUT); i++) { XOUT[i] = 1.0 / (1.0 + exp(-XOUT[i])); } } while(0)

#define E_VALUE                     2.7182818284590451
#define SQRT_E_VALUE                1.6487212707001282

typedef enum {
    LORA_NONE, LORA_TANH, LORA_SIGM
} lora_act;

typedef struct {
    int32_t head_size;
    int32_t vocab_size;
    int32_t n_embd;
    int32_t n_layer;
    int32_t n_head;
    int32_t w_lora_r;
    int32_t a_lora_r;
    int32_t g_lora_r;
    int32_t v_lora_r;
} rwkv_config;

typedef struct {
    const float *ln1_weight             ;
    const float *ln1_bias               ;
    const float *ln2_weight             ;
    const float *ln2_bias               ;
    const float *att_x_r                ;
    const float *att_x_w                ;
    const float *att_x_k                ;
    const float *att_x_v                ;
    const float *att_x_a                ;
    const float *att_x_g                ;
    const float *att_w0                 ;
    const float *att_r_k                ;
    const float *att_w1_T               ;
    const float *att_w2_T               ;
    const float *att_a1_T               ;
    const float *att_a2_T               ;
    const float *att_a0                 ;
    const float *att_g1_T               ;
    const float *att_g2_T               ;
    const float *att_v2_T               ;
    const float *att_v1_T               ;
    const float *att_v0                 ;
    const float *att_k_k                ;
    const float *att_k_a                ;
    const float *att_receptance_weight  ;
    const float *att_key_weight         ;
    const float *att_value_weight       ;
    const float *att_output_weight      ;
    const float *att_ln_x_weight        ;
    const float *att_ln_x_bias          ;
    const float *ffn_x_k                ;
    const float *ffn_key_weight         ;
    const float *ffn_value_weight       ;
} block_weights;

typedef struct {
    float *raw;
    const float *emb_weight;
    const float *blocks_0_ln0_weight;
    const float *blocks_0_ln0_bias;
    block_weights *blocks;
    const float *ln_out_weight;
    const float *ln_out_bias;
    const float *head_weight;
} rwkv_weights;

typedef struct {
    const char *str;
    int id;
} TokenIndex;

typedef struct {
    const char **vocab;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
} rwkv_tokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    float temperature;
    float top_p;
    float presence_penalty;
    float frequency_penalty;
    int *occurrence;
} rwkv_sampler;

#endif
