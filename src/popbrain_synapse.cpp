#include "popbrain_synapse.h"

#include <vector>
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

#if defined(POPBRAIN_USE_CUDA)
    float* d_centroids;
    float* d_sorted_keys;
    float* d_sorted_values;
    int32_t* d_offsets;
    float* d_query;
    float* d_sims;
#endif
};

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

    size_t cur_offset = ctx->keys.size();
    size_t added_elems = (size_t)num_transitions * dim;
    ctx->keys.resize(cur_offset + added_elems);
    ctx->values.resize(cur_offset + added_elems);

    std::memcpy(&ctx->keys[cur_offset], keys, added_elems * sizeof(float));
    std::memcpy(&ctx->values[cur_offset], values, added_elems * sizeof(float));

    for (int32_t i = 0; i < num_transitions; ++i) {
        normalize_vector(&ctx->keys[cur_offset + (size_t)i * dim], dim);
        normalize_vector(&ctx->values[cur_offset + (size_t)i * dim], dim);
        ctx->cluster_ids.push_back(0);
    }

    ctx->num_memories += num_transitions;
    ctx->ivf_built = 0;

    return 0;
}

POPBRAIN_API int32_t popbrain_build_index(
    popbrain_synapse_context* ctx,
    int32_t kmeans_iterations
) {
    if (!ctx || ctx->num_memories < ctx->config.num_clusters) {
        if (ctx) {
            ctx->sorted_keys = ctx->keys;
            ctx->sorted_values = ctx->values;
            ctx->ivf_built = 0;
        }
        return 0;
    }

    int32_t C = ctx->config.num_clusters;
    int32_t M = ctx->num_memories;
    int32_t dim = ctx->config.hidden_dim;

    // 1. Initialize random centroids from stored keys
    ctx->centroids.resize(C * dim);
    for (int32_t c = 0; c < C; ++c) {
        int32_t pick = (c * (M / C)) % M;
        std::memcpy(&ctx->centroids[c * dim], &ctx->keys[pick * dim], dim * sizeof(float));
    }

    // 2. Spherical k-means clustering iterations
    std::vector<int32_t> counts(C, 0);
    for (int32_t iter = 0; iter < kmeans_iterations; ++iter) {
        std::fill(counts.begin(), counts.end(), 0);

        for (int32_t m = 0; m < M; ++m) {
            const float* k_ptr = &ctx->keys[m * dim];
            float best_sim = -2.0f;
            int32_t best_c = 0;

            for (int32_t c = 0; c < C; ++c) {
                float sim = dot_product(k_ptr, &ctx->centroids[c * dim], dim);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_c = c;
                }
            }
            ctx->cluster_ids[m] = best_c;
            counts[best_c]++;
        }

        // Recompute centroids
        std::vector<float> new_centroids(C * dim, 0.0f);
        for (int32_t m = 0; m < M; ++m) {
            int32_t c = ctx->cluster_ids[m];
            const float* k_ptr = &ctx->keys[m * dim];
            for (int32_t d = 0; d < dim; ++d) {
                new_centroids[c * dim + d] += k_ptr[d];
            }
        }
        for (int32_t c = 0; c < C; ++c) {
            if (counts[c] > 0) {
                normalize_vector(&new_centroids[c * dim], dim);
            } else {
                std::memcpy(&new_centroids[c * dim], &ctx->keys[(rand() % M) * dim], dim * sizeof(float));
            }
        }
        ctx->centroids = new_centroids;
    }

    // 3. Sort memories by cluster_id for contiguous slice access
    struct Item {
        int32_t cluster;
        int32_t orig_idx;
    };
    std::vector<Item> items(M);
    for (int32_t i = 0; i < M; ++i) {
        items[i] = { ctx->cluster_ids[i], i };
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.cluster < b.cluster;
    });

    ctx->sorted_keys.resize(M * dim);
    ctx->sorted_values.resize(M * dim);
    for (int32_t i = 0; i < M; ++i) {
        int32_t orig = items[i].orig_idx;
        std::memcpy(&ctx->sorted_keys[i * dim], &ctx->keys[orig * dim], dim * sizeof(float));
        std::memcpy(&ctx->sorted_values[i * dim], &ctx->values[orig * dim], dim * sizeof(float));
    }

    // 4. Compute cluster offsets
    ctx->offsets.assign(C + 1, 0);
    for (int32_t i = 0; i < M; ++i) {
        ctx->offsets[items[i].cluster + 1]++;
    }
    for (int32_t c = 0; c < C; ++c) {
        ctx->offsets[c + 1] += ctx->offsets[c];
    }

    ctx->ivf_built = 1;
    return 0;
}

POPBRAIN_API int32_t popbrain_modulate_hidden(
    popbrain_synapse_context* ctx,
    float* hidden_state,
    popbrain_diagnostics* diag
) {
    if (!ctx || !hidden_state) {
        if (diag) {
            diag->fired = 0;
            diag->max_similarity = 0.0f;
            diag->margin = 0.0f;
            diag->probed_candidates = 0;
            diag->sparsity_ratio = 0.0f;
        }
        return 0;
    }

    if (ctx->num_memories == 0) {
        if (diag) {
            diag->fired = 0;
            diag->max_similarity = 0.0f;
            diag->margin = 0.0f;
            diag->probed_candidates = 0;
        }
        if (ctx->config.enable_spike_sparsity) {
            float spk_th = (ctx->config.spike_threshold > 0.0f) ? ctx->config.spike_threshold : 0.04f;
            int32_t zeroed_count = 0;
            int32_t dim = ctx->config.hidden_dim;
            for (int32_t d = 0; d < dim; ++d) {
                if (std::abs(hidden_state[d]) < spk_th) {
                    hidden_state[d] = 0.0f;
                    zeroed_count++;
                }
            }
            if (diag) diag->sparsity_ratio = (float)zeroed_count / (float)dim;
        } else {
            if (diag) diag->sparsity_ratio = 0.0f;
        }
        return 0;
    }

    int32_t dim = ctx->config.hidden_dim;
    std::vector<float> q(dim);
    double h_norm = 0.0;
    for (int32_t i = 0; i < dim; ++i) {
        h_norm += hidden_state[i] * hidden_state[i];
    }
    float norm_val = std::sqrt(std::max(1e-12, h_norm));
    for (int32_t i = 0; i < dim; ++i) {
        q[i] = hidden_state[i] / norm_val;
    }

    // 1. Gather candidate keys (either via IVF routing or flat scan)
    std::vector<int32_t> candidate_clusters;
    if (ctx->ivf_built && ctx->config.num_clusters > 1) {
        int32_t C = ctx->config.num_clusters;
        std::vector<std::pair<float, int32_t>> c_sims(C);
        for (int32_t c = 0; c < C; ++c) {
            c_sims[c] = { dot_product(q.data(), &ctx->centroids[c * dim], dim), c };
        }
        std::partial_sort(c_sims.begin(), c_sims.begin() + ctx->config.nprobe, c_sims.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        for (int32_t p = 0; p < ctx->config.nprobe; ++p) {
            candidate_clusters.push_back(c_sims[p].second);
        }
    }

    // 2. Evaluate candidate similarities and track top-1, top-2
    float best_s1 = -1.0f;
    float best_s2 = -1.0f;
    int32_t best_cand_idx = -1;
    int32_t total_cands = 0;

    std::vector<std::pair<float, int32_t>> active_sims;

    auto check_cand = [&](int32_t idx, const float* k_ptr) {
        float sim = dot_product(q.data(), k_ptr, dim);
        total_cands++;
        if (sim > best_s1) {
            best_s2 = best_s1;
            best_s1 = sim;
            best_cand_idx = idx;
        } else if (sim > best_s2) {
            best_s2 = sim;
        }
        if (sim >= ctx->config.inhibition_threshold) {
            active_sims.push_back({ sim, idx });
        }
    };

    if (ctx->ivf_built) {
        for (int32_t c : candidate_clusters) {
            int32_t s_idx = ctx->offsets[c];
            int32_t e_idx = ctx->offsets[c + 1];
            for (int32_t i = s_idx; i < e_idx; ++i) {
                check_cand(i, &ctx->sorted_keys[i * dim]);
            }
        }
    } else {
        for (int32_t i = 0; i < ctx->num_memories; ++i) {
            check_cand(i, &ctx->keys[i * dim]);
        }
    }

    float margin = (best_s2 >= 0.0f) ? (best_s1 - best_s2) : 1.0f;

    if (diag) {
        diag->max_similarity = best_s1;
        diag->margin = margin;
        diag->probed_candidates = total_cands;
    }

    // 3. GABAergic Lateral Inhibition & Differential Margin Check
    if (best_s1 >= ctx->config.inhibition_threshold && margin >= ctx->config.margin_threshold && !active_sims.empty()) {
        float sum_weights = 0.0f;
        for (auto& item : active_sims) {
            float w = item.first - ctx->config.inhibition_threshold;
            sum_weights += w;
            item.first = w;
        }

        std::vector<float> modulated_dir(dim, 0.0f);
        const float* v_src = ctx->ivf_built ? ctx->sorted_values.data() : ctx->values.data();

        for (const auto& item : active_sims) {
            float norm_w = item.first / sum_weights;
            const float* v_ptr = &v_src[item.second * dim];
            for (int32_t d = 0; d < dim; ++d) {
                modulated_dir[d] += norm_w * v_ptr[d];
            }
        }

        // Apply shift: h = h + alpha * direction * ||h||
        float alpha = ctx->config.modulation_strength;
        for (int32_t d = 0; d < dim; ++d) {
            hidden_state[d] += alpha * modulated_dir[d] * norm_val;
        }

        if (diag) diag->fired = 1;
    } else {
        if (diag) diag->fired = 0;
    }

    // 4. SNN Event-Driven Spike Activation Gating (Inducing 80%+ structural sparsity)
    if (ctx->config.enable_spike_sparsity) {
        float spk_th = (ctx->config.spike_threshold > 0.0f) ? ctx->config.spike_threshold : 0.04f;
        int32_t zeroed_count = 0;
        for (int32_t d = 0; d < dim; ++d) {
            if (std::abs(hidden_state[d]) < spk_th) {
                hidden_state[d] = 0.0f;
                zeroed_count++;
            }
        }
        if (diag) {
            diag->sparsity_ratio = (float)zeroed_count / (float)dim;
        }
    } else {
        if (diag) {
            diag->sparsity_ratio = 0.0f;
        }
    }

    return (diag && diag->fired) ? 1 : 0;
}

POPBRAIN_API int32_t popbrain_save_checkpoint(
    const popbrain_synapse_context* ctx,
    const char* filepath
) {
    if (!ctx || !filepath) return -1;
    std::ofstream out(filepath, std::ios::binary);
    if (!out) return -1;

    // Header: Magic "POPSYN01"
    const char magic[8] = {'P','O','P','S','Y','N','0','1'};
    out.write(magic, 8);
    out.write((const char*)&ctx->config, sizeof(popbrain_config));
    out.write((const char*)&ctx->num_memories, sizeof(int32_t));
    out.write((const char*)&ctx->ivf_built, sizeof(int32_t));

    size_t k_bytes = ctx->keys.size() * sizeof(float);
    out.write((const char*)ctx->keys.data(), k_bytes);
    out.write((const char*)ctx->values.data(), k_bytes);

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
    if (std::memcmp(magic, "POPSYN01", 8) != 0) return -2;

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
    popbrain_build_index(ctx, 3);

    return 0;
}

POPBRAIN_API int32_t popbrain_get_memory_count(const popbrain_synapse_context* ctx) {
    return ctx ? ctx->num_memories : 0;
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
