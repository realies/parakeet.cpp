#pragma once
#include <cmath>
#include <cstdint>
namespace pk {
// argmax over a[0..n): first index of the max (matches torch.max tie-break).
// Optional token mask: n entries, with the final entry (blank) always allowed.
// nullptr preserves the unmasked path, also used for TDT duration logits.
inline int decode_argmax(const float* a, int n, const uint8_t* allowed = nullptr) {
    if (allowed) {
        int best = -1;
        for (int i = 0; i < n; ++i)
            if ((allowed[i] || i == n - 1) && (best < 0 || a[i] > a[best])) best = i;
        return best;
    }
    int best = 0; float bv = a[0];
    for (int i = 1; i < n; ++i) if (a[i] > bv) { bv = a[i]; best = i; }
    return best;
}
// NeMo rescaled max_prob confidence over a[0..n) at index k:
//   conf = (N*p_max - 1)/(N - 1), p_max = softmax(a)[k]. Stable softmax.
// With a mask, excluded logits have zero probability; N remains the head size.
inline float decode_max_prob_conf(const float* a, int n, int k,
                                  const uint8_t* allowed = nullptr) {
    const float mx = a[decode_argmax(a, n, allowed)];
    double denom = 0.0;
    for (int i = 0; i < n; ++i)
        if (!allowed || allowed[i] || i == n - 1) denom += std::exp((double)a[i] - (double)mx);
    const double p_max = std::exp((double)a[k] - (double)mx) / denom;
    const double N = (double)n;
    return (float)((N * p_max - 1.0) / (N - 1.0));
}
} // namespace pk
