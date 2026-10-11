#include "popbrain_synapse.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "======================================================" << std::endl;
    std::cout << "PopBrain Synaptic Memory Engine C++ Self-Test Suite" << std::endl;
    std::cout << "======================================================" << std::endl;

    popbrain_config cfg;
    cfg.hidden_dim = 128;
    cfg.num_clusters = 4;
    cfg.nprobe = 2;
    cfg.inhibition_threshold = 0.85f;
    cfg.margin_threshold = 0.02f;
    cfg.modulation_strength = 0.20f;
    cfg.use_cuda = 0;
    cfg.device_id = 0;

    std::cout << "1. Initializing popbrain context (dim=" << cfg.hidden_dim << ", clusters=" << cfg.num_clusters << ")..." << std::endl;
    popbrain_synapse_context* ctx = popbrain_init(&cfg);
    assert(ctx != nullptr);
    assert(popbrain_get_memory_count(ctx) == 0);

    // Ingest 50 synthetic facts
    std::cout << "2. Ingesting 50 synthetic factual transitions..." << std::endl;
    std::vector<float> sample_k(cfg.hidden_dim);
    std::vector<float> sample_v(cfg.hidden_dim);
    std::vector<float> target_key_42(cfg.hidden_dim);

    for (int i = 0; i < 50; ++i) {
        for (int d = 0; d < cfg.hidden_dim; ++d) {
            sample_k[d] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            sample_v[d] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }
        if (i == 42) {
            target_key_42 = sample_k;
        }
        popbrain_ingest_transition(ctx, sample_k.data(), sample_v.data());
    }

    assert(popbrain_get_memory_count(ctx) == 50);
    std::cout << "   Stored count: " << popbrain_get_memory_count(ctx) << " (PASSED)" << std::endl;

    // Build Voronoi index
    std::cout << "3. Building Spherical Voronoi Index (IVF-4)..." << std::endl;
    popbrain_build_index(ctx, 4);
    std::cout << "   Index build completed (PASSED)" << std::endl;

    // Test query matching target fact #42 (should fire)
    std::cout << "4. Testing exact matching query (Fact #42)..." << std::endl;
    std::vector<float> query_h = target_key_42;

    popbrain_diagnostics diag;
    int fired = popbrain_modulate_hidden(ctx, query_h.data(), &diag);
    std::cout << "   Fired: " << fired << " | Max Sim: " << diag.max_similarity 
              << " | Margin: " << diag.margin << " | Probed: " << diag.probed_candidates << std::endl;
    assert(fired == 1);
    assert(diag.max_similarity > 0.99f);
    std::cout << "   Target query recall test (PASSED)" << std::endl;

    // Test distractor query (orthogonal / noise, should NOT fire)
    std::cout << "5. Testing distractor query (Orthogonal noise)..." << std::endl;
    std::vector<float> distractor_h(cfg.hidden_dim, 0.0f);
    distractor_h[0] = 100.0f; // Spike on single dimension, low correlation
    popbrain_diagnostics diag_dist;
    int fired_dist = popbrain_modulate_hidden(ctx, distractor_h.data(), &diag_dist);
    std::cout << "   Fired: " << fired_dist << " | Max Sim: " << diag_dist.max_similarity 
              << " | Margin: " << diag_dist.margin << std::endl;
    assert(fired_dist == 0);
    std::cout << "   Distractor cross-talk suppression test (PASSED)" << std::endl;

    // Test binary checkpoint serialization
    std::cout << "6. Testing checkpoint save & load..." << std::endl;
    const char* ckpt_path = "test_memory.synapse";
    assert(popbrain_save_checkpoint(ctx, ckpt_path) == 0);

    popbrain_synapse_context* loaded_ctx = popbrain_init(&cfg);
    assert(popbrain_load_checkpoint(loaded_ctx, ckpt_path) == 0);
    assert(popbrain_get_memory_count(loaded_ctx) == 50);
    std::cout << "   Checkpoint persistence verification (PASSED)" << std::endl;

    // Test SNN Event-Driven Activation Sparsity Gating
    std::cout << "7. Testing SNN Event-Driven Activation Sparsity Gating..." << std::endl;
    cfg.enable_spike_sparsity = 1;
    cfg.spike_threshold = 0.50f;
    popbrain_synapse_context* snn_ctx = popbrain_init(&cfg);
    std::vector<float> dense_act(cfg.hidden_dim);
    for (int d = 0; d < cfg.hidden_dim; ++d) {
        dense_act[d] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Uniform in [-1, 1]
    }
    popbrain_diagnostics diag_snn;
    popbrain_modulate_hidden(snn_ctx, dense_act.data(), &diag_snn);
    std::cout << "   Observed SNN Activation Sparsity: " << (diag_snn.sparsity_ratio * 100.0f) << "% zeros" << std::endl;
    assert(diag_snn.sparsity_ratio >= 0.40f); // Sparsity induced
    std::cout << "   SNN event-driven spike gating (PASSED)" << std::endl;

    popbrain_free(snn_ctx);
    popbrain_free(ctx);
    popbrain_free(loaded_ctx);

    std::cout << "======================================================" << std::endl;
    std::cout << "ALL POPBRAIN SYNAPSE C++ TESTS PASSED SUCCESSFULLY!" << std::endl;
    std::cout << "======================================================" << std::endl;
    return 0;
}
