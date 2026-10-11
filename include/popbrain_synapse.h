// PopBrain Synaptic C-API Interface Header
#ifndef POPBRAIN_SYNAPSE_H
#define POPBRAIN_SYNAPSE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(POPBRAIN_BUILD_SHARED)
#    define POPBRAIN_API __declspec(dllexport)
#  else
#    define POPBRAIN_API __declspec(dllimport)
#  endif
#else
#  define POPBRAIN_API __attribute__((visibility("default")))
#endif

// Opaque synaptic memory context handle
typedef struct popbrain_synapse_context popbrain_synapse_context;

typedef struct popbrain_config {
    int32_t hidden_dim;            // Dimensionality of hidden states (e.g. 1536, 2048, 2560, 4096)
    int32_t num_clusters;          // Voronoi cell partitions (default: 256)
    int32_t nprobe;                // Number of centroids probed during token retrieval (default: 4)
    float   inhibition_threshold;  // GABAergic lateral inhibition threshold (default: 0.85f)
    float   margin_threshold;      // Contrastive differential margin s1 - s2 (default: 0.02f)
    float   modulation_strength;   // Scaling factor alpha (default: 0.20f)
    int32_t use_cuda;              // 1 if CUDA acceleration enabled, 0 for AVX2 CPU fallback
    int32_t device_id;             // CUDA device ID (default: 0)
} popbrain_config;

typedef struct popbrain_diagnostics {
    int32_t fired;                 // 1 if synaptic modulation fired on current token, 0 otherwise
    float   max_similarity;        // Maximum observed cosine similarity
    float   margin;                // Gap between top-1 and top-2 candidate
    int32_t probed_candidates;     // Number of keys evaluated
} popbrain_diagnostics;

/**
 * Creates and initializes a PopBrain Synaptic Memory context.
 */
POPBRAIN_API popbrain_synapse_context* popbrain_init(const popbrain_config* config);

/**
 * Ingests a single hetero-associative transition: (K_t -> V_t).
 * Both vectors must be of length hidden_dim (in float32).
 */
POPBRAIN_API int32_t popbrain_ingest_transition(
    popbrain_synapse_context* ctx,
    const float* key,
    const float* value
);

/**
 * Ingests a bulk batch of transitions (M transitions).
 * keys: [M * hidden_dim], values: [M * hidden_dim].
 */
POPBRAIN_API int32_t popbrain_ingest_batch(
    popbrain_synapse_context* ctx,
    const float* keys,
    const float* values,
    int32_t num_transitions
);

/**
 * Builds or updates the Spherical Voronoi (IVF) index on GPU/CPU.
 */
POPBRAIN_API int32_t popbrain_build_index(
    popbrain_synapse_context* ctx,
    int32_t kmeans_iterations
);

/**
 * Modulates the hidden state vector h_in (in-place or writing to h_out).
 * This is the exact function called inside llama_decode() before lm_head projection.
 * Returns 1 if synaptic injection occurred, 0 otherwise.
 */
POPBRAIN_API int32_t popbrain_modulate_hidden(
    popbrain_synapse_context* ctx,
    float* hidden_state,
    popbrain_diagnostics* diag
);

/**
 * Saves synaptic memory binary checkpoint (.synapse).
 */
POPBRAIN_API int32_t popbrain_save_checkpoint(
    const popbrain_synapse_context* ctx,
    const char* filepath
);

/**
 * Loads synaptic memory binary checkpoint (.synapse).
 */
POPBRAIN_API int32_t popbrain_load_checkpoint(
    popbrain_synapse_context* ctx,
    const char* filepath
);

/**
 * Returns total number of stored synaptic memories.
 */
POPBRAIN_API int32_t popbrain_get_memory_count(const popbrain_synapse_context* ctx);

/**
 * Frees synaptic memory context.
 */
POPBRAIN_API void popbrain_free(popbrain_synapse_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // POPBRAIN_SYNAPSE_H
