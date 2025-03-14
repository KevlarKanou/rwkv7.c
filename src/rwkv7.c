#include "rwkv7.h"    

// operators
void mat_mul_vec(float *xout, const float *x, const float *w, int x_len, int xout_len) {
    // W (d,n) @ x (n,) -> xout (d,)
    int d = xout_len;
    int n = x_len;
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void vec_mul_mat(float *xout, const float *x, const float *w, int x_len, int xout_len) {
    // x (n,) @ W (n,d) -> xout (d,)
    int d = xout_len;
    int n = x_len;
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += w[j * d + i] * x[j];
        }
        xout[i] = sum;
    }
}

void layer_norm(float *xout, const float *x, const float *weight, const float *bias, int size) {
    float x_mean = 0.0;
    for (int i = 0; i < size; i++) { x_mean += x[i]; }
    x_mean /= size;

    float x_var = 0.0;
    for (int i = 0; i < size; i++) { x_var += (x[i] - x_mean) * (x[i] - x_mean); }
    x_var /= size;

    for (int i = 0; i < size; i++) {
        xout[i] = ((x[i] - x_mean) / sqrt(x_var + 1e-5f)) * weight[i] + bias[i];
    }
}

void group_norm(float *xout, const float *x, const float *weight, const float *bias, int group_num, int group_size) {
    float x_mean[group_num], x_var[group_num];
    for (int i = 0; i < group_num; i++) {
        x_mean[i] = 0.0;
        for (int j = 0; j < group_size; j++) {
            int x_idx = IDX(i, j, 0, group_size, 1);
            x_mean[i] += x[x_idx];
        }
        x_mean[i] /= group_size;
        
        x_var[i] = 0.0;
        for (int j = 0; j < group_size; j++) {
            int x_idx = IDX(i, j, 0, group_size, 1);
            x_var[i] += (x[x_idx] - x_mean[i]) * (x[x_idx] - x_mean[i]);
        }
        x_var[i] /= group_size;
    }

    for (int i = 0; i < group_num; i++) {
        for (int j = 0; j < group_size; j++) {
            int x_idx = IDX(i, j, 0, group_size, 1);
            xout[x_idx] = ((x[x_idx] - x_mean[i]) / sqrt(x_var[i] + 64e-5f)) * weight[x_idx] + bias[x_idx];
        }
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

void lerp(float *xout, const float *x, const float *last_x, const float *mu, int size) {
    for (int i = 0; i < size; i++) {
        xout[i] = x[i] + mu[i] * (last_x[i] - x[i]);
    }
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
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

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
    float w_tanh_[c->w_lora_r], w_sigmoid_[c->n_embd];
    VECxMAT(w_tanh_, xw, bw->att_w1);                                                   // xw @ Ww1
    VECTANH(w_tanh_);                                                                   // np.tanh(xw @ Ww1)
    VECxMAT(w_sigmoid_, w_tanh_, bw->att_w2);                                           // np.tanh(xw @ Ww1) @ Ww2
    VECADD(w_sigmoid_, w_sigmoid_, bw->att_w0);                                         // np.tanh(xw @ Ww1) @ Ww2 + w_bias
    VECSIGM(w_sigmoid_);                                                                // sigmoid(...)
    for (int i = 0; i < c->n_embd; i++) { w[i] = exp(-w_sigmoid_[i] / SQRT_E_VALUE); }  // exp(...)

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
        float xv_mul_Wv1[c->v_lora_r];
        float v_sigmoid_[c->n_embd];
        VECxMAT(xv_mul_Wv1, xv, bw->att_v1);                                            // xv @ Wv1
        VECxMAT(v_sigmoid_, xv_mul_Wv1, bw->att_v2);                                    // xv @ Wv1 @ Wv2
        VECADD(v_sigmoid_, v_sigmoid_, bw->att_v0);                                     // xv @ Wv1 @ Wv2 + v_bias
        VECSIGM(v_sigmoid_);                                                            // sigmoid(...)
        for (int i = 0; i < ARRLEN(v); i++) { v[i] += (v0[i] - v[i]) * v_sigmoid_[i]; } // (v0 - v) * sigmoid(...)
    }

    // a = sigmoid(xa @ Wa1 @ Wa2 + a_bias)
    float a[c->n_embd];
    float xa_mul_Wa1[c->a_lora_r];
    VECxMAT(xa_mul_Wa1, xa, bw->att_a1);    // xa @ Wa1
    VECxMAT(a, xa_mul_Wa1, bw->att_a2);     // xa @ Wa1 @ Wa2
    VECADD(a, a, bw->att_a0);               // xa @ Wa1 @ Wa2 + a_bias
    VECSIGM(a);                             // sigmoid(...)

    // g = sigmoid(xg @ Wg1) @ Wg2
    float g[c->n_embd];
    float g_sigmoid_[c->g_lora_r];
    VECxMAT(g_sigmoid_, xg, bw->att_g1);    // xg @ Wg1
    VECSIGM(g_sigmoid_);                    // sigmoid(...)
    VECxMAT(g, g_sigmoid_, bw->att_g2);     // sigmoid(...) * Wg2

    // kk = k * k_k
    float kk[c->n_embd];
    HADAMARD(kk, k, bw->att_k_k);

    // k += k * (a-1) * k_a
    for (int i = 0; i < ARRLEN(k); i++) { k[i] += k[i] * (a[i] - 1) * bw->att_k_a[i]; }

    // r,w,k,v,kk,a,r_k = [i.reshape(N_HEAD, HEAD_SIZE, 1) for i in [r,w,k,v,kk,a,r_k]]
    // kk /= np.maximum(np.linalg.norm(kk, axis=1,keepdims=1), 1e-12)
    // TODO
    for (int i = 0; i < c->n_head; i++) {
        float kk_norm = 0.0;
        for (int j = 0; j < c->head_size; j++) { kk_norm += SQUARE(kk[i * c->head_size + j]); }
        kk_norm = sqrt(kk_norm);
        for (int j = 0; j < c->head_size; j++) { kk[i * c->head_size + j] /= MAXIMUM(kk_norm, 1e-12f); }
    }

    /*---S = S * w.mT - S @ kk * (kk*a).mT + v * k.mT---*/
    // tmp0: S * w.mT -> X[i,j,k] = S[i,j,k] * w[i,k,0]
    // tmp1: S @ kk * (kk*a).mT = X[i,j,k]=(∑(m=0 -> 63)​S[i,j,m]*kk[i,m,0])×(kk[i,k,0]×a[i,k,0])
    // tmp2: v * k.mT -> X[i,j,k] = v[i,j,0] * k[i,k,0]
    do {
        // S * w.mT
        // TODO
        float state_mul_w_mT[c->n_head * c->head_size * c->head_size];
        for (int i = 0; i < c->n_head; i++) {
            for (int j = 0; j < c->head_size; j++) {
                for (int k = 0; k < c->head_size; k++) {
                    // S[i,j,k] * w[i,k,0]
                    int state_idx = IDX(i, j, k, c->head_size, c->head_size);
                    int w_idx = IDX(i, k, 0, c->head_size, 1);
                    state_mul_w_mT[state_idx] = state[state_idx] * w[w_idx];
                }
            }
        }
        // S @ kk
        float state_mul_kk[c->n_head * c->head_size];
        for (int i = 0; i < c->n_head; i++) {
            int state_offset = IDX(i, 0, 0, c->head_size, c->head_size);
            int kk_offset = IDX(i, 0, 0, c->head_size, 1);
            mat_mul_vec(state_mul_kk + kk_offset, kk + kk_offset, state + state_offset, c->head_size, c->head_size);
        }
        // kk * a
        float kk_mul_a[c->n_head * c->head_size];
        HADAMARD(kk_mul_a, kk, a);
        // tmp1
        float tmp1[c->n_head * c->head_size * c->head_size];
        for (int i = 0; i < c->n_head; i++) {
            for (int j = 0; j < c->head_size; j++) {
                for (int k = 0; k < c->head_size; k++) {
                    // temp1[i,j,0] * temp2.mT[i,0,k] = temp1[i,j,0] * temp2[i,k,0]
                    float val1 = state_mul_kk[IDX(i, j, 0, c->head_size, 1)];
                    float val2 = kk_mul_a[IDX(i, k, 0, c->head_size, 1)];
                    tmp1[IDX(i, j, k, c->head_size, c->head_size)] = val1 * val2;
                }
            }
        }
        // v * k.mT
        float v_mul_k_mT[c->n_head * c->head_size * c->head_size];
        for (int i = 0; i < c->n_head; i++) {
            for (int j = 0; j < c->head_size; j++) {
                for (int m = 0; m < c->head_size; m++) {
                    // v[i,j,0] * k[i,m,0]
                    int v_idx = IDX(i, j, 0, c->head_size, 1);
                    int k_idx = IDX(i, m, 0, c->head_size, 1);
                    v_mul_k_mT[IDX(i, j, m, c->head_size, c->head_size)] = v[v_idx] * k[k_idx];
                }
            }
        }
        // final result
        VECSUB_L(state, state_mul_w_mT, tmp1, ARRLEN(state_mul_w_mT));
        VECADD_L(state, state, v_mul_k_mT, ARRLEN(v_mul_k_mT));
    } while (0);
    /*---END: S = S * w.mT - S @ kk * (kk*a).mT + v * k.mT---*/

    // y = S @ r 
    float y[c->n_head * c->head_size];
    for (int i = 0; i < c->n_head; i++) {
        int state_offset = IDX(i, 0, 0, c->head_size, c->head_size);
        int r_offset = IDX(i, 0, 0, c->head_size, 1);
        mat_mul_vec(y + r_offset, r + r_offset, state + state_offset, c->head_size, c->head_size);
    }

    // y = group_norm(y, ln_w, ln_b)
    group_norm(y, y, bw->att_ln_x_weight, bw->att_ln_x_bias, c->n_head, c->head_size);

    // y += ((r * k * r_k).sum(axis=1,keepdims=1) * v).flatten()
    // TODO
    float y_sum_[c->n_head];
    for (int i = 0; i < c->n_head; i++) {
        y_sum_[i] = 0.0;
        for (int j = 0; j < c->head_size; j++) {
            int idx = IDX(i, j, 0, c->head_size, 1);
            y_sum_[i] += r[idx] * k[idx] * bw->att_r_k[idx];
        }
    }
    for (int i = 0; i < c->n_head; i++) {
        for (int j = 0; j < c->head_size; j++) {
            int idx = IDX(i, j, 0, c->head_size, 1);
            y[idx] += y_sum_[i] * v[idx];
        }
    }

    // dx = Wo @ (y * g)
    float y_mul_g[c->n_embd];
    HADAMARD(y_mul_g, y, g);
    mat_mul_vec(dx, y_mul_g, bw->att_output_weight, ARRLEN(y_mul_g), c->n_embd);

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
    layer_norm(x, x, w->blocks_0_ln0_weight, w->blocks_0_ln0_bias, ARRLEN(x));

    float x_[c->n_embd];
    float v0[c->n_embd]; v0[0] = NAN;
    float dx[c->n_embd];

    for (int i = 0; i < c->n_layer; i++) {
        layer_norm(x_, x, w->blocks[i].ln1_weight, w->blocks[i].ln1_bias, ARRLEN(x_));

        int last_x_offset = IDX(i, 0, 0, 2, c->n_embd);
        int state_offset = i * c->n_head * c->head_size * c->head_size;
        time_mixing(dx, x_, v0, model_state[0] + last_x_offset, model_state[1] + state_offset, w->blocks + i, c);
        VECADD(x, x, dx);

        layer_norm(x_, x, w->blocks[i].ln2_weight, w->blocks[i].ln2_bias, ARRLEN(x_));
        last_x_offset = IDX(i, 1, 0, 2, c->n_embd);
        channel_mixing(dx, x_, model_state[0] + last_x_offset, w->blocks + i, c);
        VECADD(x, x, dx);
    }

    layer_norm(x, x, w->ln_out_weight, w->ln_out_bias, ARRLEN(x));
    mat_mul_vec(logits, x, w->head_weight, ARRLEN(x), c->vocab_size);
}

// load and free
void load_model(const char *model_path, rwkv_config *c, rwkv_weights *w) {
    FILE *fp = fopen(model_path, "r");
    assert(fp);

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
    if (fread(&header, sizeof(header), 1, fp) != 1) { exit(EXIT_FAILURE); }

    assert(header.magic_number == 0x00632E37766B7772);
    assert(header.quant == 0);

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
    if (fread(w->raw, sizeof(uint8_t), raw_weights_size, fp) != raw_weights_size) { exit(EXIT_FAILURE); }
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

    assert((ptr - w->raw) * sizeof(float) == raw_weights_size);
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
    assert(sorted_vocab_idx == t->vocab_size);

    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    t->max_token_length = strlen(t->sorted_vocab[t->vocab_size-1].str);
}

void free_tokenizer(rwkv_tokenizer *t) {
    free(t->sorted_vocab);
}

// main
void error_usage(char *argv[]) {
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
    if ((msg == NULL) | (model_path == NULL)) { error_usage(argv); }
    if (sampler.temperature         < 0.0) { sampler.temperature        = 0.0; }
    if (sampler.top_p               < 0.0) { sampler.top_p              = 0.0; }
    if (sampler.presence_penalty    < 0.0) { sampler.presence_penalty   = 0.0; }
    if (sampler.frequency_penalty   < 0.0) { sampler.frequency_penalty  = 0.0; }
    srand(seed);

    printf("Hello, RWKV!, seed: %u\n\n", seed);
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
        if (next_token == 0) { break; }

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
