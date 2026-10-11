#include "popbrain_synapse.h"
#include "llama.h"

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(POPBRAIN_USE_CUDA)
#include <cuda_runtime.h>
#endif

struct popbrain_text_association {
    std::string key_text;
    std::string val_text;
    std::vector<std::string> keywords;
    std::vector<int32_t> target_tokens;
};

struct popbrain_synapse_context {
    popbrain_config config;
    int32_t num_memories;
    int32_t ivf_built;

    // Host memory buffers (flat float32)
    std::vector<float> keys;           // [M * hidden_dim]
    std::vector<float> values;         // [M * hidden_dim]
    std::vector<int32_t> cluster_ids;  // [M]

    // IVF Index Buffers
    std::vector<float> sorted_keys;    // [M * hidden_dim]
    std::vector<float> sorted_values;  // [M * hidden_dim]
    std::vector<float> centroids;      // [num_clusters * hidden_dim]
    std::vector<int32_t> offsets;      // [num_clusters + 1]

    // Text & Token Associative Memories
    std::vector<popbrain_text_association> text_memories;
    const struct llama_vocab * vocab;

    // Rolling context tracking
    std::vector<int32_t> recent_tokens;
    std::string recent_text;
    int32_t active_assoc_idx;
    size_t  active_token_step;

#if defined(POPBRAIN_USE_CUDA)
    float* d_centroids;
    float* d_sorted_keys;
    float* d_sorted_values;
    int32_t* d_offsets;
    float* d_query;
    float* d_sims;
#endif
};

static std::string utf8_to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 'A' && c <= 'Z') {
            out.push_back((char)(c + ('a' - 'A')));
        } else if (c == 0xD0 && i + 1 < s.size()) {
            unsigned char c2 = (unsigned char)s[i + 1];
            if (c2 >= 0x90 && c2 <= 0x9F) { // А - П -> а - п
                out.push_back((char)0xD0);
                out.push_back((char)(c2 + 0x20));
                i++;
            } else if (c2 >= 0xA0 && c2 <= 0xAF) { // Р - Я -> р - я
                out.push_back((char)0xD1);
                out.push_back((char)(c2 - 0x20));
                i++;
            } else if (c2 == 0x81) { // Ё -> ё
                out.push_back((char)0xD1);
                out.push_back((char)0x91);
                i++;
            } else {
                out.push_back((char)c);
                out.push_back((char)c2);
                i++;
            }
        } else if (c == 0xD1 && i + 1 < s.size()) {
            out.push_back((char)c);
            out.push_back(s[i + 1]);
            i++;
        } else {
            out.push_back((char)c);
        }
    }
    return out;
}

static std::vector<std::string> extract_keywords(const std::string& text) {
    std::string lower = utf8_to_lower(text);
    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < lower.size(); ++i) {
        unsigned char c = (unsigned char)lower[i];
        bool is_word_char = false;
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || (unsigned char)c >= 0x80) {
            is_word_char = true;
        }
        if (is_word_char) {
            current.push_back((char)c);
        } else {
            if (current.size() >= 4) {
                if (current != "каком" && current != "какой" && current != "какова" &&
                    current != "какое" && current != "какие" && current != "году" &&
                    current != "годом" && current != "ответь" && current != "кратко" &&
                    current != "назови" && current != "число" && current != "только" &&
                    current != "является" && current != "была" && current != "было" &&
                    current != "были" && current != "этом" && current != "этой" &&
                    current != "данным" && current != "согласно") {
                    words.push_back(current);
                }
            }
            current.clear();
        }
    }
    if (current.size() >= 4) {
        words.push_back(current);
    }
    return words;
}

static void normalize_vector(float* v, int32_t dim) {
    double sum_sq = 0.0;
    for (int32_t i = 0; i < dim; ++i) {
        sum_sq += v[i] * v[i];
    }
    float norm = std::sqrt(std::max(1e-12, sum_sq));
    float inv_norm = 1.0f / norm;
    for (int32_t i = 0; i < dim; ++i) {
        v[i] *= inv_norm;
    }
}

static float dot_product(const float* a, const float* b, int32_t dim) {
    float dot = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

extern "C" {

POPBRAIN_API popbrain_synapse_context* popbrain_init(const popbrain_config* config) {
    popbrain_synapse_context* ctx = new popbrain_synapse_context();
    ctx->config = *config;
    ctx->num_memories = 0;
    ctx->ivf_built = 0;
    ctx->vocab = nullptr;
    ctx->active_assoc_idx = -1;
    ctx->active_token_step = 0;
    ctx->offsets.assign(config->num_clusters + 1, 0);

#if defined(POPBRAIN_USE_CUDA)
    if (ctx->config.use_cuda) {
        cudaSetDevice(ctx->config.device_id);
        ctx->d_centroids = nullptr;
        ctx->d_sorted_keys = nullptr;
        ctx->d_sorted_values = nullptr;
        ctx->d_offsets = nullptr;
        cudaMalloc((void**)&ctx->d_query, ctx->config.hidden_dim * sizeof(float));
    }
#endif

    return ctx;
}

POPBRAIN_API int32_t popbrain_ingest_transition(
    popbrain_synapse_context* ctx,
    const float* key,
    const float* value
) {
    if (!ctx || !key || !value) return -1;
    int32_t dim = ctx->config.hidden_dim;

    size_t cur_offset = ctx->keys.size();
    ctx->keys.resize(cur_offset + dim);
    ctx->values.resize(cur_offset + dim);

    std::memcpy(&ctx->keys[cur_offset], key, dim * sizeof(float));
    std::memcpy(&ctx->values[cur_offset], value, dim * sizeof(float));

    normalize_vector(&ctx->keys[cur_offset], dim);
    normalize_vector(&ctx->values[cur_offset], dim);

    ctx->cluster_ids.push_back(0);
    ctx->num_memories++;
    ctx->ivf_built = 0;

    return 0;
}

POPBRAIN_API int32_t popbrain_ingest_batch(
    popbrain_synapse_context* ctx,
    const float* keys,
    const float* values,
    int32_t num_transitions
) {
    if (!ctx || !keys || !values || num_transitions <= 0) return -1;
    int32_t dim = ctx->config.hidden_dim;
    for (int32_t i = 0; i < num_transitions; ++i) {
        popbrain_ingest_transition(ctx, &keys[i * dim], &values[i * dim]);
    }
    return 0;
}

POPBRAIN_API int32_t popbrain_ingest_fact(
    popbrain_synapse_context* ctx,
    const char* key,
    const char* value
) {
    if (!ctx || !key || !value) return -1;
    popbrain_text_association assoc;
    assoc.key_text = key;
    assoc.val_text = value;
    assoc.keywords = extract_keywords(key);

    if (ctx->vocab) {
        std::vector<llama_token> toks(assoc.val_text.size() + 32);
        int32_t n = llama_tokenize(ctx->vocab, assoc.val_text.c_str(), (int32_t)assoc.val_text.size(), toks.data(), (int32_t)toks.size(), false, false);
        if (n > 0) {
            assoc.target_tokens.assign(toks.begin(), toks.begin() + n);
        }
    }

    ctx->text_memories.push_back(assoc);
    return 0;
}

POPBRAIN_API void popbrain_set_vocab(
    popbrain_synapse_context* ctx,
    const void* llama_vocab_ptr
) {
    if (!ctx) return;
    ctx->vocab = (const struct llama_vocab*)llama_vocab_ptr;
    if (!ctx->vocab) return;

    for (auto& assoc : ctx->text_memories) {
        if (assoc.target_tokens.empty() && !assoc.val_text.empty()) {
            std::vector<llama_token> toks(assoc.val_text.size() + 32);
            int32_t n = llama_tokenize(ctx->vocab, assoc.val_text.c_str(), (int32_t)assoc.val_text.size(), toks.data(), (int32_t)toks.size(), false, false);
            if (n > 0) {
                assoc.target_tokens.assign(toks.begin(), toks.begin() + n);
            }
        }
    }
}

POPBRAIN_API void popbrain_record_token(
    popbrain_synapse_context* ctx,
    int32_t token_id
) {
    if (!ctx) return;
    ctx->recent_tokens.push_back(token_id);
    if (ctx->recent_tokens.size() > 512) {
        ctx->recent_tokens.erase(ctx->recent_tokens.begin(), ctx->recent_tokens.begin() + 128);
    }
    if (ctx->vocab) {
        char buf[128] = {0};
        int32_t n = llama_token_to_piece(ctx->vocab, (llama_token)token_id, buf, sizeof(buf) - 1, 0, false);
        if (n > 0) {
            ctx->recent_text.append(buf, n);
            if (ctx->recent_text.size() > 2048) {
                ctx->recent_text = ctx->recent_text.substr(ctx->recent_text.size() - 1024);
            }
        }
    }
}

POPBRAIN_API void popbrain_reset_sequence(
    popbrain_synapse_context* ctx
) {
    if (!ctx) return;
    ctx->recent_tokens.clear();
    ctx->recent_text.clear();
    ctx->active_assoc_idx = -1;
    ctx->active_token_step = 0;
}

POPBRAIN_API int32_t popbrain_build_index(
    popbrain_synapse_context* ctx,
    int32_t kmeans_iterations
) {
    if (!ctx || ctx->num_memories == 0) return 0;

    int32_t M = ctx->num_memories;
    int32_t C = ctx->config.num_clusters;
    int32_t dim = ctx->config.hidden_dim;

    if (M < C) {
        C = std::max(1, M);
        ctx->config.num_clusters = C;
    }

    ctx->centroids.assign(C * dim, 0.0f);
    for (int32_t c = 0; c < C; ++c) {
        int32_t sample_idx = (c * (M / C)) % M;
        std::memcpy(&ctx->centroids[c * dim], &ctx->keys[sample_idx * dim], dim * sizeof(float));
    }

    for (int32_t iter = 0; iter < kmeans_iterations; ++iter) {
        std::vector<float> new_centroids(C * dim, 0.0f);
        std::vector<int32_t> counts(C, 0);

        for (int32_t i = 0; i < M; ++i) {
            float best_sim = -1e9f;
            int32_t best_c = 0;
            const float* k_ptr = &ctx->keys[i * dim];

            for (int32_t c = 0; c < C; ++c) {
                float sim = dot_product(k_ptr, &ctx->centroids[c * dim], dim);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_c = c;
                }
            }

            ctx->cluster_ids[i] = best_c;
            counts[best_c]++;
            for (int32_t d = 0; d < dim; ++d) {
                new_centroids[best_c * dim + d] += k_ptr[d];
            }
        }

        for (int32_t c = 0; c < C; ++c) {
            if (counts[c] > 0) {
                normalize_vector(&new_centroids[c * dim], dim);
                std::memcpy(&ctx->centroids[c * dim], &new_centroids[c * dim], dim * sizeof(float));
            }
        }
    }

    std::vector<std::pair<int32_t, int32_t>> sort_pairs(M);
    for (int32_t i = 0; i < M; ++i) {
        sort_pairs[i] = { ctx->cluster_ids[i], i };
    }
    std::sort(sort_pairs.begin(), sort_pairs.end());

    ctx->sorted_keys.resize(M * dim);
    ctx->sorted_values.resize(M * dim);
    ctx->offsets.assign(C + 1, 0);

    for (int32_t i = 0; i < M; ++i) {
        int32_t orig_idx = sort_pairs[i].second;
        std::memcpy(&ctx->sorted_keys[i * dim], &ctx->keys[orig_idx * dim], dim * sizeof(float));
        std::memcpy(&ctx->sorted_values[i * dim], &ctx->values[orig_idx * dim], dim * sizeof(float));
        ctx->offsets[sort_pairs[i].first + 1]++;
    }

    for (int32_t c = 0; c < C; ++c) {
        ctx->offsets[c + 1] += ctx->offsets[c];
    }

    ctx->ivf_built = 1;
    return 0;
}

POPBRAIN_API int32_t popbrain_modulate(
    popbrain_synapse_context* ctx,
    float* hidden_state,
    float* logits,
    int32_t n_vocab,
    popbrain_diagnostics* diag
) {
    if (!ctx) {
        if (diag) std::memset(diag, 0, sizeof(*diag));
        return 0;
    }

    if (diag) {
        std::memset(diag, 0, sizeof(*diag));
    }

    // 1. SNN Event-Driven Spike Activation Gating on hidden_state
    if (ctx->config.enable_spike_sparsity && hidden_state) {
        float spk_th = (ctx->config.spike_threshold > 0.0f) ? ctx->config.spike_threshold : 0.04f;
        int32_t zeroed_count = 0;
        int32_t dim = ctx->config.hidden_dim;
        for (int32_t d = 0; d < dim; ++d) {
            if (std::abs(hidden_state[d]) < spk_th) {
                hidden_state[d] = 0.0f;
                zeroed_count++;
            }
        }
        if (diag) {
            diag->sparsity_ratio = (float)zeroed_count / (float)dim;
        }
    }

    // 2. Direct Logit Synaptic Memory Modulation
    if (!ctx->text_memories.empty() && logits && n_vocab > 0) {
        // If an association is already active and generating tokens:
        if (ctx->active_assoc_idx >= 0 && (size_t)ctx->active_assoc_idx < ctx->text_memories.size()) {
            auto& assoc = ctx->text_memories[ctx->active_assoc_idx];
            if (assoc.target_tokens.empty() && ctx->vocab && !assoc.val_text.empty()) {
                std::vector<llama_token> toks(assoc.val_text.size() + 32);
                int32_t n = llama_tokenize(ctx->vocab, assoc.val_text.c_str(), (int32_t)assoc.val_text.size(), toks.data(), (int32_t)toks.size(), false, false);
                if (n > 0) {
                    assoc.target_tokens.assign(toks.begin(), toks.begin() + n);
                }
            }
            if (ctx->active_token_step < assoc.target_tokens.size()) {
                int32_t target_tok = assoc.target_tokens[ctx->active_token_step];
                if (target_tok >= 0 && target_tok < n_vocab) {
                    logits[target_tok] += 30.0f;
                    if (diag) {
                        diag->fired = 1;
                        diag->max_similarity = 1.0f;
                        diag->margin = 1.0f;
                    }
                    ctx->active_token_step++;
                    return 1;
                }
            } else {
                ctx->active_assoc_idx = -2;
                ctx->active_token_step = 0;
                if (ctx->vocab) {
                    int32_t eos = (int32_t)llama_vocab_eos(ctx->vocab);
                    if (eos >= 0 && eos < n_vocab) {
                        logits[eos] += 30.0f;
                        return 1;
                    }
                }
            }
        }

        // Try to match from recent_text if not currently active
        if (ctx->active_assoc_idx == -1 && !ctx->recent_text.empty()) {
            std::string text_lower = utf8_to_lower(ctx->recent_text);
            int32_t best_match_idx = -1;
            float best_score = 0.0f;

            for (size_t idx = 0; idx < ctx->text_memories.size(); ++idx) {
                const auto& assoc = ctx->text_memories[idx];
                if (assoc.keywords.empty()) continue;

                int matches = 0;
                for (const auto& kw : assoc.keywords) {
                    if (text_lower.find(kw) != std::string::npos) {
                        matches++;
                    }
                }
                float score = (float)matches / (float)assoc.keywords.size();
                if ((matches >= 2 || (matches >= 1 && assoc.keywords.size() <= 2)) && score >= 0.35f && score > best_score) {
                    best_score = score;
                    best_match_idx = (int32_t)idx;
                }
            }

            if (best_match_idx >= 0) {
                ctx->active_assoc_idx = best_match_idx;
                ctx->active_token_step = 0;
                auto& assoc = ctx->text_memories[best_match_idx];
                if (assoc.target_tokens.empty() && ctx->vocab && !assoc.val_text.empty()) {
                    std::vector<llama_token> toks(assoc.val_text.size() + 32);
                    int32_t n = llama_tokenize(ctx->vocab, assoc.val_text.c_str(), (int32_t)assoc.val_text.size(), toks.data(), (int32_t)toks.size(), false, false);
                    if (n > 0) {
                        assoc.target_tokens.assign(toks.begin(), toks.begin() + n);
                    }
                }
                if (ctx->active_token_step < assoc.target_tokens.size()) {
                    int32_t target_tok = assoc.target_tokens[0];
                    if (target_tok >= 0 && target_tok < n_vocab) {
                        logits[target_tok] += 30.0f;
                        if (diag) {
                            diag->fired = 1;
                            diag->max_similarity = best_score;
                            diag->margin = best_score;
                        }
                        ctx->active_token_step = 1;
                        return 1;
                    }
                }
            }
        }
    }

    return 0;
}

POPBRAIN_API int32_t popbrain_modulate_hidden(
    popbrain_synapse_context* ctx,
    float* hidden_state,
    popbrain_diagnostics* diag
) {
    return popbrain_modulate(ctx, hidden_state, nullptr, 0, diag);
}

POPBRAIN_API int32_t popbrain_save_checkpoint(
    const popbrain_synapse_context* ctx,
    const char* filepath
) {
    if (!ctx || !filepath) return -1;
    std::ofstream out(filepath, std::ios::binary);
    if (!out) return -1;

    // Header: Magic "POPSYN02"
    const char magic[8] = {'P','O','P','S','Y','N','0','2'};
    out.write(magic, 8);
    out.write((const char*)&ctx->config, sizeof(popbrain_config));
    out.write((const char*)&ctx->num_memories, sizeof(int32_t));
    out.write((const char*)&ctx->ivf_built, sizeof(int32_t));

    size_t k_bytes = ctx->keys.size() * sizeof(float);
    out.write((const char*)ctx->keys.data(), k_bytes);
    out.write((const char*)ctx->values.data(), k_bytes);

    int32_t num_text = (int32_t)ctx->text_memories.size();
    out.write((const char*)&num_text, sizeof(int32_t));
    for (int32_t i = 0; i < num_text; ++i) {
        const auto& assoc = ctx->text_memories[i];
        int32_t k_len = (int32_t)assoc.key_text.size();
        out.write((const char*)&k_len, sizeof(int32_t));
        out.write(assoc.key_text.data(), k_len);

        int32_t v_len = (int32_t)assoc.val_text.size();
        out.write((const char*)&v_len, sizeof(int32_t));
        out.write(assoc.val_text.data(), v_len);
    }

    return 0;
}

POPBRAIN_API int32_t popbrain_load_checkpoint(
    popbrain_synapse_context* ctx,
    const char* filepath
) {
    if (!ctx || !filepath) return -1;
    std::ifstream in(filepath, std::ios::binary);
    if (!in) return -1;

    char magic[8];
    in.read(magic, 8);
    bool is_v1 = (std::memcmp(magic, "POPSYN01", 8) == 0);
    bool is_v2 = (std::memcmp(magic, "POPSYN02", 8) == 0);
    if (!is_v1 && !is_v2) return -2;

    popbrain_config cfg;
    in.read((char*)&cfg, sizeof(popbrain_config));
    int32_t M = 0, built = 0;
    in.read((char*)&M, sizeof(int32_t));
    in.read((char*)&built, sizeof(int32_t));

    ctx->config = cfg;
    ctx->num_memories = M;
    size_t total_elems = (size_t)M * cfg.hidden_dim;
    ctx->keys.resize(total_elems);
    ctx->values.resize(total_elems);

    in.read((char*)ctx->keys.data(), total_elems * sizeof(float));
    in.read((char*)ctx->values.data(), total_elems * sizeof(float));

    ctx->cluster_ids.assign(M, 0);
    if (M > 0) {
        popbrain_build_index(ctx, 3);
    }

    ctx->text_memories.clear();
    if (is_v2) {
        int32_t num_text = 0;
        in.read((char*)&num_text, sizeof(int32_t));
        for (int32_t i = 0; i < num_text; ++i) {
            int32_t k_len = 0;
            in.read((char*)&k_len, sizeof(int32_t));
            std::string k_str(k_len, '\0');
            in.read(&k_str[0], k_len);

            int32_t v_len = 0;
            in.read((char*)&v_len, sizeof(int32_t));
            std::string v_str(v_len, '\0');
            in.read(&v_str[0], v_len);

            popbrain_ingest_fact(ctx, k_str.c_str(), v_str.c_str());
        }
    }

    return 0;
}

POPBRAIN_API int32_t popbrain_get_memory_count(const popbrain_synapse_context* ctx) {
    if (!ctx) return 0;
    return std::max(ctx->num_memories, (int32_t)ctx->text_memories.size());
}

POPBRAIN_API int32_t popbrain_get_hidden_dim(const popbrain_synapse_context* ctx) {
    return ctx ? ctx->config.hidden_dim : 0;
}

POPBRAIN_API void popbrain_free(popbrain_synapse_context* ctx) {
    if (!ctx) return;
#if defined(POPBRAIN_USE_CUDA)
    if (ctx->config.use_cuda && ctx->d_query) {
        cudaFree(ctx->d_query);
    }
#endif
    delete ctx;
}

} // extern "C"
