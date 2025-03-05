#ifndef __RWKV7_H__
#define __RWKV7_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "rwkv_vocab_v20230424.h"

#define ARRLEN(X)                   (int)(sizeof(X)/sizeof(X[0]))
#define IDX(I, J, K, DIM2, DIM3)    ((I) * (DIM2) * (DIM3) + (J) * (DIM3) + (K))
#define SQUARE(X)                   ((X) * (X))
#define MAXIMUM(A, B)               ((A > B) ? A : B)
#define RELU(X)                     MAXIMUM(X, 0)
#define IS_NAN(X)                   (!((X) == (X)))
#define MATxVEC(XOUT, X, W)         mat_mul_vec(XOUT, X, W, ARRLEN(X), ARRLEN(XOUT))
#define VECxMAT(XOUT, X, W)         vec_mul_mat(XOUT, X, W, ARRLEN(X), ARRLEN(XOUT))
#define VECADD_L(XOUT, A, B, L)     do { for (int i = 0; i < (L); i++) { XOUT[i] = A[i] + B[i]; } } while(0)
#define VECADD(XOUT, A, B)          VECADD_L(XOUT, A, B, ARRLEN(XOUT))
#define VECSUB_L(XOUT, A, B, L)     do { for (int i = 0; i < (L); i++) { XOUT[i] = A[i] - B[i]; } } while(0)
#define VECSUB(XOUT, A, B)          VECSUB_L(XOUT, A, B, ARRLEN(XOUT))
#define HADAMARD(XOUT, A, B)        do { for (int i = 0; i < ARRLEN(XOUT); i++) { XOUT[i] = A[i] * B[i]; } } while(0)
#define VECTANH(XOUT)               do { for (int i = 0; i < ARRLEN(XOUT); i++) { XOUT[i] = tanh(XOUT[i]); } } while(0)
#define VECSIGM(XOUT)               do { for (int i = 0; i < ARRLEN(XOUT); i++) { XOUT[i] = 1.0 / (1.0 + exp(-XOUT[i])); } } while(0)
#define LERP(XOUT, X, LAST_X, MU)   lerp(XOUT, X, LAST_X, MU, ARRLEN(XOUT))

#define E_VALUE                     2.7182818284590451
#define SQRT_E_VALUE                1.6487212707001282

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
    const float *att_w1                 ;
    const float *att_w2                 ;
    const float *att_a1                 ;
    const float *att_a2                 ;
    const float *att_a0                 ;
    const float *att_g1                 ;
    const float *att_g2                 ;
    const float *att_v2                 ;
    const float *att_v1                 ;
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

#endif
