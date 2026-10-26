#include "decode_common.hpp"

#include <cstdio>
#include <cstdint>
#include <limits>
#include <type_traits>

// Also run against the original two-argument argmax for the regression proof.
template<class F>
static int choose(F argmax, const float* logits, int n, const uint8_t* allowed) {
    if constexpr (std::is_invocable_v<F, const float*, int, const uint8_t*>)
        return argmax(logits, n, allowed);
    else
        return argmax(logits, n);
}

template<class F>
static float confidence(F conf, const float* logits, int n, int k, const uint8_t* allowed) {
    if constexpr (std::is_invocable_v<F, const float*, int, int, const uint8_t*>)
        return conf(logits, n, k, allowed);
    else
        return conf(logits, n, k);
}

int main() {
    // Four tokens + blank, followed by two duration logits (outside the mask).
    float logits[] = {100.0f, 3.0f, 5.0f, 2.0f, 4.0f, 200.0f, 300.0f};
    const uint8_t allowed[] = {0, 1, 1, 0, 0};
    const int best = choose(&pk::decode_argmax, logits, 5, allowed);
    if (best != 2) {
        std::fprintf(stderr, "masked greedy: expected allowed token 2, got %d\n", best);
        return 1;
    }
    if (choose(&pk::decode_argmax, logits, 5, nullptr) != 0) return 1;
    if (pk::decode_argmax(logits + 5, 2) != 1 || logits[5] != 200.0f || logits[6] != 300.0f) return 1;

    logits[4] = 6.0f;
    if (choose(&pk::decode_argmax, logits, 5, allowed) != 4) return 1;
    const uint8_t blank_only[] = {0, 0, 0, 0, 0};
    if (choose(&pk::decode_argmax, logits, 5, blank_only) != 4) return 1;

    // Ties choose the first allowed token, even when blank ties too.
    logits[1] = logits[2] = logits[4] = 5.0f;
    if (choose(&pk::decode_argmax, logits, 5, allowed) != 1) return 1;
    if (std::fabs(confidence(&pk::decode_max_prob_conf, logits, 5, 1, allowed) - 1.0f/6.0f) > 1e-6f) return 1;
    if (confidence(&pk::decode_max_prob_conf, logits, 5, 4, blank_only) != 1.0f) return 1;
    for (float& value : logits) value = -std::numeric_limits<float>::infinity();
    if (choose(&pk::decode_argmax, logits, 5, allowed) != 1) return 1;
    std::fprintf(stderr, "masked greedy: PASS (allowed best, blank, clear, ties, durations)\n");
    return 0;
}
