#include "rwkv7.h"    

// operators
#ifdef AVX
static float _avx_horizontal_sum(__m256 v) {
    __m128 v1 = _mm256_extractf128_ps(v, 0);
    __m128 v2 = _mm256_extractf128_ps(v, 1);
    v1 = _mm_add_ps(v1, v2);
    v1 = _mm_hadd_ps(v1, v1);
    v1 = _mm_hadd_ps(v1, v1);
    return _mm_cvtss_f32(v1);
}

void _avx_vec_add(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 sum_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(xout + i, sum_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] + b[i]; }
}

void _avx_vec_sub(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 sum_vec = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(xout + i, sum_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] - b[i]; }
}

void _avx_hadamard(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 sum_vec = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(xout + i, sum_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] * b[i]; }
}
#endif

#ifdef NEON
static float _neon_horizontal_sum(float32x4_t v) {
    float32x2_t v1 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    v1 = vpadd_f32(v1, v1);
    return vget_lane_f32(v1, 0);
}

void _neon_vec_add(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 4; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t sum_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(xout + i, sum_vec); 
    }
    for (; i < len; i++) { xout[i] = a[i] + b[i]; }
}

void _neon_vec_sub(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 4; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t sum_vec = vsubq_f32(a_vec, b_vec);
        vst1q_f32(xout + i, sum_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] - b[i]; }
}

void _neon_hadamard(float *xout, const float *a, const float *b, int len) {
    int i;
    for (i = 0; i <= len - 4; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t prod_vec = vmulq_f32(a_vec, b_vec);
        vst1q_f32(xout + i, prod_vec);
    }
    for (; i < len; i++) { xout[i] = a[i] * b[i]; }
}
#endif

void mat_mul_vec(float *xout, const float *x, const float *w, int x_len, int xout_len) {
    // W (d,n) @ x (n,) -> xout (d,) 
    int d = xout_len;
    int n = x_len;
#if defined(AVX)
    for (int i = 0; i < d; i++) {
        __m256 sum_vec = _mm256_setzero_ps();
        int j;
        for (j = 0; j <= n - 8; j += 8) {
            __m256 w_vec = _mm256_loadu_ps(w + i * n + j);
            __m256 x_vec = _mm256_loadu_ps(x + j);
            sum_vec = _mm256_fmadd_ps(w_vec, x_vec, sum_vec);
        }
        float val = _avx_horizontal_sum(sum_vec);

        for (; j < n; j++) { val += w[i * n + j] * x[j]; }
        xout[i] = val;
    }
#elif defined(NEON)
    for (int i = 0; i < d; i++) {
        float32x4_t sum_vec = vdupq_n_f32(0);
        int j;
        for (j = 0; j <= n - 4; j += 4) {
            float32x4_t w_vec = vld1q_f32(w + i * n + j);
            float32x4_t x_vec = vld1q_f32(x + j);
            sum_vec = vfmaq_f32(sum_vec, w_vec, x_vec);
        }
        float val = _neon_horizontal_sum(sum_vec);

        for (; j < n; j++) { val += w[i * n + j] * x[j]; }
        xout[i] = val;
    }
#else
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}

void vec_mul_mat(float *xout, const float *x, const float *w, int x_len, int xout_len) {
    // x (n,) @ W (n,d) -> xout (d,)
    int d = xout_len;
    int n = x_len;
#if defined(AVX)
    for (int i = 0; i < d; i++) {
        __m256 sum_vec = _mm256_setzero_ps();
        int j;
        for (j = 0; j <= n - 8; j += 8) {
#ifdef AVX2
            __m256i base = _mm256_set1_epi32(j * d + i);
            __m256i offsets = _mm256_mullo_epi32(
                _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                _mm256_set1_epi32(d)
            );
            __m256i indices = _mm256_add_epi32(base, offsets);
            __m256 w_vec = _mm256_i32gather_ps(w, indices, 4);
#else
            float w_col[8];
            for (int k = 0; k < 8; k++) { w_col[k] = w[(j + k) * d + i]; }
            __m256 w_vec = _mm256_loadu_ps(w_col);
#endif
            __m256 x_vec = _mm256_loadu_ps(x + j);
            sum_vec = _mm256_fmadd_ps(w_vec, x_vec, sum_vec);
        }
        float val = _avx_horizontal_sum(sum_vec);

        for (; j < n; j++) { val += w[j * d + i] * x[j]; }
        xout[i] = val;
    }
#elif defined(NEON)
    for (int i = 0; i < d; i++) {
        float32x4_t sum_vec = vdupq_n_f32(0);
        int j;
        for (j = 0; j <= n - 4; j += 4) {
            float w_col[4];
            for (int k = 0; k < 4; k++) { w_col[k] = w[(j + k) * d + i]; }
            float32x4_t w_vec = vld1q_f32(w_col);
            float32x4_t x_vec = vld1q_f32(x + j);
            sum_vec = vfmaq_f32(sum_vec, w_vec, x_vec);
        }
        float val = _neon_horizontal_sum(sum_vec);

        for (; j < n; j++) { val += w[j * d + i] * x[j]; }
        xout[i] = val;
    }
#else
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += w[j * d + i] * x[j];
        }
        xout[i] = sum;
    }
#endif
}

void vec_out_product(float *xout, const float *a, const float *b, int vec_len) {
#if defined(AVX)
    for (int i = 0; i < vec_len; i++) {
        const __m256 a_vec = _mm256_set1_ps(a[i]);
        int j = 0;
        for (; j <= vec_len - 8; j += 8) {
            const __m256 b_vec = _mm256_loadu_ps(b + j);
            const __m256 result = _mm256_mul_ps(a_vec, b_vec);
            _mm256_storeu_ps(&xout[i * vec_len + j], result);
        }
        
        for (; j < vec_len; j++) { xout[i * vec_len + j] = a[i] * b[j]; }
    }
#elif defined(NEON)
    for (int i = 0; i < vec_len; i++) {
        const float32x4_t a_vec = vdupq_n_f32(a[i]);
        int j = 0;
        for (; j <= vec_len - 4; j += 4) {
            const float32x4_t b_vec = vld1q_f32(b + j);
            const float32x4_t result = vmulq_f32(a_vec, b_vec);
            vst1q_f32(&xout[i * vec_len + j], result);
        }
        
        for (; j < vec_len; j++) { xout[i * vec_len + j] = a[i] * b[j]; }
    }
#else
    for (int i = 0; i < vec_len; i++) {
        for (int j = 0; j < vec_len; j++) { xout[i * vec_len + j] = a[i] * b[j]; }
    }
#endif
}

void layer_norm(float *xout, const float *x, const float *weight, const float *bias, int size, float sqrt_bias) {
    float x_mean = 0.0;
    for (int i = 0; i < size; i++) { x_mean += x[i]; }
    x_mean /= size;

    float x_var = 0.0;
    for (int i = 0; i < size; i++) { x_var += (x[i] - x_mean) * (x[i] - x_mean); }
    x_var /= size;

    for (int i = 0; i < size; i++) {
        xout[i] = ((x[i] - x_mean) / sqrt(x_var + sqrt_bias)) * weight[i] + bias[i];
    }
}

void softmax(float *xout, const float *x, float temp, int size) {
    float max = 0.0;
    for (int i = 0; i < size; i++) {
        if (x[i] > max) { max = x[i]; }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        xout[i] = exp((x[i] - max) / temp);
        sum += xout[i];
    }
    for (int i = 0; i < size; i++) { xout[i] /= sum; }
}

void lerp(float *xout, const float *x, const float *last_x, const float *mu, int x_len) {
#if defined(AVX)
    int i;
    for (i = 0; i <= x_len - 8; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 last_x_vec = _mm256_loadu_ps(last_x + i);
        __m256 mu_vec = _mm256_loadu_ps(mu + i);

        __m256 xout_vec = _mm256_sub_ps(last_x_vec, x_vec);
        xout_vec = _mm256_fmadd_ps(mu_vec, xout_vec, x_vec);
        _mm256_storeu_ps(xout + i, xout_vec);
    }
    for (; i < x_len; i++) {
        xout[i] = x[i] + mu[i] * (last_x[i] - x[i]);
    }
#elif defined(NEON)
    int i;
    for (i = 0; i <= x_len - 4; i += 4) {
        float32x4_t x_vec = vld1q_f32(x + i);
        float32x4_t last_x_vec = vld1q_f32(last_x + i);
        float32x4_t mu_vec = vld1q_f32(mu + i);

        float32x4_t xout_vec = vsubq_f32(last_x_vec, x_vec);
        xout_vec = vfmaq_f32(x_vec, mu_vec, xout_vec);
        vst1q_f32(xout + i, xout_vec);
    }
    for (; i < x_len; i++) {
        xout[i] = x[i] + mu[i] * (last_x[i] - x[i]);
    }
#else
    for (int i = 0; i < x_len; i++) {
        xout[i] = x[i] + mu[i] * (last_x[i] - x[i]);
    }
#endif
}

void lora(float *xout, const float *x, const float *weight_1, const float *weight_2, int x_len, int lora_rank, lora_act func) {
    float tmp[lora_rank];
    vec_mul_mat(tmp, x, weight_1, x_len, lora_rank);
    switch (func) {
        case LORA_NONE: break;
        case LORA_TANH: VECTANH(tmp); break;
        case LORA_SIGM: VECSIGM(tmp); break;
        default: ERR(1, "unknown lora activation function");
    }
    vec_mul_mat(xout, tmp, weight_2, lora_rank, x_len);
}

// utils for tokenizer
// temporarily use the implementation from llama2.c
int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(rwkv_tokenizer *t, char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    ERR(text == NULL, "cannot encode NULL text");

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    int str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    if (text[0] == '\0') {
        tokens[(*n_tokens)++] = 1;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1) {
                // this merge pair exists in vocab! record its score and position
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
}

// sampler
int compare_probs(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_logits(float* logits, rwkv_config *c, rwkv_sampler *s) {
    int next = 0;
    // apply repetition penalty to logits
    for (int i = 0; i < c->vocab_size; i++) {
        logits[i] -= s->presence_penalty + s->occurrence[i] * s->frequency_penalty;
    }
    if (s->temperature != 0.0f) {
        softmax(logits, logits, s->temperature, c->vocab_size);
    }

    ProbIndex sorted_probs[c->vocab_size];
    for (int i = 0; i < c->vocab_size; i++) {
        sorted_probs[i].index = i;
        sorted_probs[i].prob = logits[i];
    }
    qsort(sorted_probs, c->vocab_size, sizeof(ProbIndex), compare_probs);

    if (s->temperature != 0.0f) {
        // calculate cutoff length
        int cutoff_len = 0;
        float cumulative_probs = 0;
        for (; cutoff_len < c->vocab_size; cutoff_len++) {
            cumulative_probs += sorted_probs[cutoff_len].prob;
            if (s->top_p <= cumulative_probs) {
                cutoff_len++;
                break;
            }
        }
        // roll a die!
        float die = (float)rand() / (float)RAND_MAX * cumulative_probs;
        cumulative_probs = 0.0f;
        for (int i = 0; i < cutoff_len; i++) {
            cumulative_probs += sorted_probs[i].prob;
            if (die <= cumulative_probs) {
                next = sorted_probs[i].index;
                break;
            }
        }
    }
    else {
        next = sorted_probs[0].index;
    }

    // maintain repetition penalty
    for (int i = 0; i < c->vocab_size; i++) { s->occurrence[i] *= s->presence_penalty; }
    s->occurrence[next] += 1;

    return next;
}

// RWKV block
void time_mixing(float *dx, const float *x, float *v0, float *last_x, float *state, block_weights *bw, rwkv_config *c) {
    float xr[c->n_embd], xw[c->n_embd], xk[c->n_embd], xv[c->n_embd], xa[c->n_embd], xg[c->n_embd];
    LERP(xr, x, last_x, bw->att_x_r);
    LERP(xw, x, last_x, bw->att_x_w);
    LERP(xk, x, last_x, bw->att_x_k);
    LERP(xv, x, last_x, bw->att_x_v);
    LERP(xa, x, last_x, bw->att_x_a);
    LERP(xg, x, last_x, bw->att_x_g);

    // r = Wr @ xr
    float r[c->n_embd];
    MATxVEC(r, xr, bw->att_receptance_weight);

    // w = np.exp(-sigmoid(np.tanh(xw @ Ww1) @ Ww2 + w_bias)/np.e**0.5)
    float w[c->n_embd];
    do {
        float w_sigmoid_[c->n_embd];
        lora(w_sigmoid_, xw, bw->att_w1, bw->att_w2, ARRLEN(xw), c->w_lora_r, LORA_TANH);   // np.tanh(xw @ Ww1) @ Ww2
        VECADD(w_sigmoid_, w_sigmoid_, bw->att_w0);                                         // np.tanh(xw @ Ww1) @ Ww2 + w_bias
        VECSIGM(w_sigmoid_);                                                                // sigmoid(...)
        for (int i = 0; i < c->n_embd; i++) { w[i] = exp(-w_sigmoid_[i] / SQRT_E_VALUE); }  // exp(...)
    } while(0); // w = np.exp(-sigmoid(np.tanh(xw @ Ww1) @ Ww2 + w_bias)/np.e**0.5)

    // k = Wk @ xk
    float k[c->n_embd];
    MATxVEC(k, xk, bw->att_key_weight);

    // v = Wv @ xv
    float v[c->n_embd];
    MATxVEC(v, xv, bw->att_value_weight);

    if (IS_NAN(v0[0])) {
        memcpy(v0, v, sizeof(float) * c->n_embd);
    }
    else {
        // v += (v0 - v) * sigmoid(xv @ Wv1 @ Wv2 + v_bias)
        float v_sigmoid_[c->n_embd];
        lora(v_sigmoid_, xv, bw->att_v1, bw->att_v2, ARRLEN(xv), c->v_lora_r, LORA_NONE);   // xv @ Wv1 @ Wv2
        VECADD(v_sigmoid_, v_sigmoid_, bw->att_v0);                                         // xv @ Wv1 @ Wv2 + v_bias
        VECSIGM(v_sigmoid_);                                                                // sigmoid(...)
        for (int i = 0; i < ARRLEN(v); i++) { v[i] += (v0[i] - v[i]) * v_sigmoid_[i]; }     // (v0 - v) * sigmoid(...)
    }

    // a = sigmoid(xa @ Wa1 @ Wa2 + a_bias)
    float a[c->n_embd];
    lora(a, xa, bw->att_a1, bw->att_a2, ARRLEN(xa), c->a_lora_r, LORA_NONE);    // xa @ Wa1 @ Wa2
    VECADD(a, a, bw->att_a0);                                                   // xa @ Wa1 @ Wa2 + a_bias
    VECSIGM(a);                                                                 // sigmoid(...)

    // g = sigmoid(xg @ Wg1) @ Wg2
    float g[c->n_embd];
    lora(g, xg, bw->att_g1, bw->att_g2, ARRLEN(xg), c->g_lora_r, LORA_SIGM);

    // kk = k * k_k
    float kk[c->n_embd];
    HADAMARD(kk, k, bw->att_k_k);

    // k += k * (a-1) * k_a
    for (int i = 0; i < ARRLEN(k); i++) { k[i] += k[i] * (a[i] - 1) * bw->att_k_a[i]; }

    // multi-head
    float y[c->n_head * c->head_size];
    for (int i = 0; i < c->n_head; i++) {
        float *head_state   = state + i * c->head_size * c->head_size;
        float *head_kk      = kk    + i * c->head_size;
        float *head_y       = y     + i * c->head_size;
        const float *head_r = r     + i * c->head_size;
        const float *head_w = w     + i * c->head_size;
        const float *head_k = k     + i * c->head_size;
        const float *head_v = v     + i * c->head_size;
        const float *head_a = a     + i * c->head_size;

        const float *ln_w   = bw->att_ln_x_weight   + i * c->head_size;
        const float *ln_b   = bw->att_ln_x_bias     + i * c->head_size;
        const float *r_k    = bw->att_r_k           + i * c->head_size;

        // kk /= np.maximum(np.linalg.norm(kk, axis=1,keepdims=1), 1e-12)
        do {
            float kk_norm = 0.0;
            for (int j = 0; j < c->head_size; j++) { kk_norm += SQUARE(head_kk[j]); }
            kk_norm = sqrt(kk_norm);
            for (int j = 0; j < c->head_size; j++) { head_kk[j] /= MAXIMUM(kk_norm, 1e-12f); }
        } while(0); // kk /= np.maximum(np.linalg.norm(kk, axis=1,keepdims=1), 1e-12)

        // RWKV v7 paper: S = S @ (diag(w) - kk @ (kk * a).mT) + v * k.mT
        // - multiply S into the parentheses, 
        // - to avoid non-contiguous memory access of column vectors in matrix computation
        // - S = S * w.mT - S @ kk * (kk * a).mT + v * k.mT
        do {
            float state_mul_kk[c->head_size];
            mat_mul_vec(state_mul_kk, head_kk, head_state, c->head_size, c->head_size); // S @ kk

            float kk_mul_a[c->head_size];
            HADAMARD(kk_mul_a, head_kk, head_a);                                        // kk * a

            float tmp[c->head_size * c->head_size];
            vec_out_product(tmp, state_mul_kk, kk_mul_a, c->head_size);                 // S @ kk * (kk*a).mT

            float v_mul_k[c->head_size * c->head_size];
            vec_out_product(v_mul_k, head_v, head_k, c->head_size);                     // v * k.mT

            for (int j = 0; j < c->head_size; j++) {
                float *state_row = head_state + j * c->head_size;
                HADAMARD_L(state_row, state_row, head_w, c->head_size);                 // S = S * w.mT
            }

            VECSUB_L(head_state, head_state, tmp, c->head_size * c->head_size);         // S -= S @ kk * (kk * a).mT
            VECADD_L(head_state, head_state, v_mul_k, c->head_size * c->head_size);     // S += v * k.mT
        } while(0); // S = S * w.mT - S @ kk * (kk * a).mT + v * k.mT

        // y = S @ r
        mat_mul_vec(head_y, head_r, head_state, c->head_size, c->head_size);

        // y = group_norm(y, ln_w, ln_b)
        layer_norm(head_y, head_y, ln_w, ln_b, c->head_size, 64e-5f);

        // y += ((r * k * r_k).sum(axis=1,keepdims=1) * v).flatten()
        do {
            float y_sum_ = 0.0f;
            for (int j = 0; j < c->head_size; j++) {
                y_sum_ += head_r[j] * head_k[j] * r_k[j];
            }
            for (int j = 0; j < c->head_size; j++) {
                head_y[j] += y_sum_ * head_v[j];
            }
        } while(0); // y += ((r * k * r_k).sum(axis=1,keepdims=1) * v).flatten()
    }   // multi-head

    // dx = Wo @ (y * g)
    do {
        float y_mul_g[c->n_embd];
        HADAMARD(y_mul_g, y, g);
        mat_mul_vec(dx, y_mul_g, bw->att_output_weight, ARRLEN(y_mul_g), c->n_embd);
    } while(0); // dx = Wo @ (y * g)

    // last_x = x
    memcpy(last_x, x, sizeof(float) * c->n_embd);
}

void channel_mixing(float *dx, const float *x, float *last_x, block_weights *bw, rwkv_config *c) {
    float k[c->n_embd * 4];
    float xk[c->n_embd];
    LERP(xk, x, last_x, bw->ffn_x_k);
    MATxVEC(k, xk, bw->ffn_key_weight);

    float v[c->n_embd];
    for (int i = 0; i < ARRLEN(k); i++) { k[i] = SQUARE(RELU(k[i])); }
    MATxVEC(v, k, bw->ffn_value_weight);
    memcpy(dx, v, sizeof(float) * c->n_embd);
    memcpy(last_x, x, sizeof(float) * c->n_embd);
}

void forward(float *logits, rwkv_config *c, rwkv_weights *w, float *model_state[], int token) {
    float x[c->n_embd];
    memcpy(x, w->emb_weight + token * c->n_embd, sizeof(float) *ARRLEN(x));
    layer_norm(x, x, w->blocks_0_ln0_weight, w->blocks_0_ln0_bias, ARRLEN(x), 1e-5f);

    float x_[c->n_embd];
    float v0[c->n_embd]; v0[0] = NAN;
    float dx[c->n_embd];

    for (int i = 0; i < c->n_layer; i++) {
        layer_norm(x_, x, w->blocks[i].ln1_weight, w->blocks[i].ln1_bias, ARRLEN(x_), 1e-5f);

        int last_x_offset = IDX(i, 0, 0, 2, c->n_embd);
        int state_offset = i * c->n_head * c->head_size * c->head_size;
        time_mixing(dx, x_, v0, model_state[0] + last_x_offset, model_state[1] + state_offset, w->blocks + i, c);
        VECADD(x, x, dx);

        layer_norm(x_, x, w->blocks[i].ln2_weight, w->blocks[i].ln2_bias, ARRLEN(x_), 1e-5f);
        last_x_offset = IDX(i, 1, 0, 2, c->n_embd);
        channel_mixing(dx, x_, model_state[0] + last_x_offset, w->blocks + i, c);
        VECADD(x, x, dx);
    }

    layer_norm(x, x, w->ln_out_weight, w->ln_out_bias, ARRLEN(x), 1e-5f);
    mat_mul_vec(logits, x, w->head_weight, ARRLEN(x), c->vocab_size);
}

// load and free
void load_model(const char *model_path, rwkv_config *c, rwkv_weights *w) {
    FILE *fp = fopen(model_path, "r");
    ERR(fp == NULL, "failed to open model file");

    // get file size
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // load header
    #pragma pack(push, 1)   // prevent compiler memory align
    struct {
        uint64_t magic_number;
        int32_t quant;
        int32_t head_size;
        int32_t n_embd;
        int32_t n_layer;
        int32_t vocab_size;
        int32_t w_lora_r;
        int32_t a_lora_r;
        int32_t g_lora_r;
        int32_t v_lora_r;
    } header;
    #pragma pack(pop)
    ERR(fread(&header, sizeof(header), 1, fp) != 1, "failed to get model header");

    ERR(header.magic_number != 0x00632E37766B7772, "invalid model magic number");
    ERR(header.quant != 0, "quantized model is not supported");

    // set model config
    c->head_size    = header.head_size;
    c->vocab_size   = header.vocab_size;
    c->n_embd       = header.n_embd;
    c->n_layer      = header.n_layer;
    c->n_head       = header.n_embd / c->head_size;
    c->w_lora_r     = header.w_lora_r;
    c->a_lora_r     = header.a_lora_r;
    c->g_lora_r     = header.g_lora_r;
    c->v_lora_r     = header.v_lora_r;

    // load weights
    w->blocks = malloc(c->n_layer * sizeof(block_weights));

    size_t raw_weights_size = file_size - sizeof(header);
    w->raw = malloc(raw_weights_size);
    ERR(fread(w->raw, sizeof(uint8_t), raw_weights_size, fp) != raw_weights_size, "failed to load model weights");
    fclose(fp);

    float *ptr = w->raw;
    w->emb_weight                   = ptr; ptr += c->vocab_size * c->n_embd     ;
    w->blocks_0_ln0_weight          = ptr; ptr += c->n_embd                     ;
    w->blocks_0_ln0_bias            = ptr; ptr += c->n_embd                     ;
    for (int i = 0; i < c->n_layer; i++) {
        block_weights *b = w->blocks + i;
        b->ln1_weight               = ptr; ptr += c->n_embd                     ;
        b->ln1_bias                 = ptr; ptr += c->n_embd                     ;
        b->ln2_weight               = ptr; ptr += c->n_embd                     ;
        b->ln2_bias                 = ptr; ptr += c->n_embd                     ;
        b->att_x_r                  = ptr; ptr += c->n_embd                     ;
        b->att_x_w                  = ptr; ptr += c->n_embd                     ;
        b->att_x_k                  = ptr; ptr += c->n_embd                     ;
        b->att_x_v                  = ptr; ptr += c->n_embd                     ;
        b->att_x_a                  = ptr; ptr += c->n_embd                     ;
        b->att_x_g                  = ptr; ptr += c->n_embd                     ;
        b->att_w0                   = ptr; ptr += c->n_embd                     ;
        b->att_r_k                  = ptr; ptr += c->n_head     * c->head_size  ;
        b->att_w1                   = ptr; ptr += c->n_embd     * c->w_lora_r   ;
        b->att_w2                   = ptr; ptr += c->w_lora_r   * c->n_embd     ;
        b->att_a1                   = ptr; ptr += c->n_embd     * c->a_lora_r   ;
        b->att_a2                   = ptr; ptr += c->a_lora_r   * c->n_embd     ;
        b->att_a0                   = ptr; ptr += c->n_embd                     ;
        b->att_g1                   = ptr; ptr += c->n_embd     * c->g_lora_r   ;
        b->att_g2                   = ptr; ptr += c->g_lora_r   * c->n_embd     ;
        if (i != 0) {
            b->att_v2               = ptr; ptr += c->v_lora_r   * c->n_embd     ;
            b->att_v1               = ptr; ptr += c->n_embd     * c->v_lora_r   ;
            b->att_v0               = ptr; ptr += c->n_embd                     ;
        }
        b->att_k_k                  = ptr; ptr += c->n_embd                     ;
        b->att_k_a                  = ptr; ptr += c->n_embd                     ;
        b->att_receptance_weight    = ptr; ptr += c->n_embd     * c->n_embd     ;
        b->att_key_weight           = ptr; ptr += c->n_embd     * c->n_embd     ;
        b->att_value_weight         = ptr; ptr += c->n_embd     * c->n_embd     ;
        b->att_output_weight        = ptr; ptr += c->n_embd     * c->n_embd     ;
        b->att_ln_x_weight          = ptr; ptr += c->n_embd                     ;
        b->att_ln_x_bias            = ptr; ptr += c->n_embd                     ;
        b->ffn_x_k                  = ptr; ptr += c->n_embd                     ;
        b->ffn_key_weight           = ptr; ptr += c->n_embd     * c->n_embd * 4 ;
        b->ffn_value_weight         = ptr; ptr += c->n_embd * 4 * c->n_embd     ;
    }
    w->ln_out_weight                = ptr; ptr += c->n_embd                     ;
    w->ln_out_bias                  = ptr; ptr += c->n_embd                     ;
    w->head_weight                  = ptr; ptr += c->n_embd     * c->vocab_size ;

    ERR((ptr - w->raw) * sizeof(float) != raw_weights_size, "failed to map model weights");
}

void free_model(rwkv_weights *w) {
    free(w->blocks);
    free(w->raw);
}

void load_tokenizer(rwkv_tokenizer *t, int vocab_size) {
    t->vocab = _vocab;
    t->vocab_size = vocab_size;

    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    int sorted_vocab_idx = 0;
    for (int i = 0; i <= t->vocab_size; i++) {
        if (t->vocab[i]) {
            t->sorted_vocab[sorted_vocab_idx].str = t->vocab[i];
            t->sorted_vocab[sorted_vocab_idx].id = i;
            sorted_vocab_idx++;
        }
    }
    ERR(sorted_vocab_idx != t->vocab_size, "failed to load vocabulary");

    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    t->max_token_length = strlen(t->sorted_vocab[t->vocab_size-1].str);
}

void free_tokenizer(rwkv_tokenizer *t) {
    free(t->sorted_vocab);
}

// main
void print_usage(char *argv[]) {
    fprintf(stderr, "Usage: %s [options] model_path\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --chat                       enable chat mode\n");
    fprintf(stderr, "  --reasoner                   enable reasoner mode\n");
    fprintf(stderr, "  -i, --input <input message>  model inference input\n");
    fprintf(stderr, "  --temperature <float>        sample temperature\n");
    fprintf(stderr, "  --top-p <float>              sample top-p\n");
    fprintf(stderr, "  --presence_penalty <float>   presence penalty\n");
    fprintf(stderr, "  --frequency_penalty <float>  frequency penalty\n");
    fprintf(stderr, "  --seed <int>                 random seed\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    rwkv_sampler sampler = { .temperature = 1.0, .top_p = 0.7, .presence_penalty = 0.1, .frequency_penalty = 0.2 };
    unsigned int seed = time(NULL);
    bool chat_mode = false, reasoner_mode = false;
    const char *msg = NULL;
    const char *model_path = NULL;

    if (argc < 2) { error_usage(argv); }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--chat") == 0) 
            { chat_mode = true; }
        else if (strcmp(argv[i], "--reasoner") == 0)
            { chat_mode = true; reasoner_mode = true; }
        else if ((strcmp(argv[i], "-i") == 0) || (strcmp(argv[i], "--input") == 0))
            { msg = argv[i + 1]; i++; }
        else if (strcmp(argv[i], "--temperature") == 0)
            { sampler.temperature = atof(argv[i + 1]); i++; }
        else if (strcmp(argv[i], "--top-p") == 0) 
            { sampler.top_p = atof(argv[i + 1]); i++; }
        else if (strcmp(argv[i], "--seed") == 0) 
            { seed = atoi(argv[i + 1]); i++; }
        else if (strcmp(argv[i], "--presence_penalty") == 0)
            { sampler.presence_penalty = atof(argv[i + 1]); i++; }
        else if (strcmp(argv[i], "--frequency_penalty") == 0)
            { sampler.frequency_penalty = atof(argv[i + 1]); i++; }
        else { model_path = argv[i]; }
    }
    if ((msg == NULL) || (model_path == NULL)) { print_usage(argv); }
    if (sampler.temperature         < 0.0) { sampler.temperature        = 0.0; }
    if (sampler.top_p               < 0.0) { sampler.top_p              = 0.0; }
    if (sampler.presence_penalty    < 0.0) { sampler.presence_penalty   = 0.0; }
    if (sampler.frequency_penalty   < 0.0) { sampler.frequency_penalty  = 0.0; }
    srand(seed);

    printf("Hello, RWKV! seed: %u\n\n", seed);
    rwkv_config config;
    rwkv_weights weights;
    rwkv_tokenizer tokenizer;

    printf("Loading model...\n");
    load_model(model_path, &config, &weights);
    printf("Model loaded!\n\n");

    printf("Loading tokenizer...\n");
    load_tokenizer(&tokenizer, VOCAB_SIZE);
    printf("Tokenizer loaded!\n\n");

    sampler.occurrence = calloc(config.vocab_size, sizeof(int));

    char *context = NULL;
    int context_len = 0;
    if (chat_mode && !reasoner_mode) {
        context_len = strlen(msg) + strlen("User: \n\nAssistant:");
        context = malloc(context_len + 1);
        sprintf(context, "User: %s\n\nAssistant:", msg);
    }
    else if (chat_mode && reasoner_mode) {
        context_len = strlen(msg) + strlen("User: \n\nAssistant:<think>");
        context = malloc(context_len + 1);
        sprintf(context, "User: %s\n\nAssistant:<think>", msg);
    }
    else {
        context_len = strlen(msg);
        context = malloc(context_len + 1);
        sprintf(context, "%s", msg);
    }

    int token_list[context_len];
    int prefilling_tokens = 0;
    encode(&tokenizer, context, token_list, &prefilling_tokens);
    free(context);

    float *model_state[2];  // init with zero
    model_state[0] = calloc(config.n_layer * 2 * config.n_embd, sizeof(float));
    model_state[1] = calloc(config.n_layer * config.n_head * config.head_size * config.head_size, sizeof(float));

    float logits[config.vocab_size];
    long start, end;
    long prefilling_time, decoding_time;

    // prefilling
    SYSTIME_MS(start);
    for (int i = 0; i < prefilling_tokens; i++) {
        forward(logits, &config, &weights, model_state, token_list[i]);
        if (!chat_mode) {
            const char *token_str = tokenizer.vocab[token_list[i]];
            printf("%s", token_str);
            fflush(stdout);
        }
    }
    SYSTIME_MS(end);
    prefilling_time = end - start;
    
    // decoding
    SYSTIME_MS(start);
    int decoding_tokens = 0;
    for (decoding_tokens = 0; decoding_tokens < 10240; decoding_tokens++) {
        int next_token = sample_logits(logits, &config, &sampler);
        if (next_token == 0) { printf("\n---Meet EOS!---\n"); break; }

        forward(logits, &config, &weights, model_state, next_token);
        const char *token_str = tokenizer.vocab[next_token];
        if (strncmp(token_str, "\n\n", 2) == 0) { break; }

        printf("%s", token_str);
        fflush(stdout);
    }
    SYSTIME_MS(end);
    decoding_time = end - start;

    printf("\n---------\n");
    printf("Prefill: %d tokens, %ld ms, %f token/s\n",
        prefilling_tokens, prefilling_time, (float)prefilling_tokens / prefilling_time * 1000);
    printf("Decode: %d tokens, %ld ms, %f token/s\n",
        decoding_tokens, decoding_time, (float)decoding_tokens / decoding_time * 1000);

    free(model_state[0]);
    free(model_state[1]);
    free_model(&weights);
    free_tokenizer(&tokenizer);
    free(sampler.occurrence);
}
