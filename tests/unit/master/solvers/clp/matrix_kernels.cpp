#include "baldes_clp_matrix_kernels.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static bool same(double a, double b) {
    if (std::isnan(a))
        return std::isnan(b);
    std::uint64_t x, y;
    std::memcpy(&x, &a, sizeof x);
    std::memcpy(&y, &b, sizeof y);
    return x == y;
}

int main() {
#if !defined(__SSE2__) || defined(BALDES_CLP_FORCE_SCALAR)
    double x = 1.;
    baldes_clp::PairedRows fallback(1, 128);
    assert(fallback.pack(&x, &x) == nullptr);
    std::puts("scalar matrix fallback selected");
    return 0;
#else
    std::mt19937_64 random(23749);
    for (int rows : {1, 3, 100, 256, 257, 1000}) {
        std::vector<double> pi(rows), weights(rows);
        for (int i = 0; i < rows; ++i) {
            pi[i] = static_cast<int>(random() % 20001) / 1000.1234;
            weights[i] = static_cast<int>(random() % 20001) / 4096.3871;
            if (i % 3 == 0)
                pi[i] = -pi[i];
        }
        baldes_clp::PairedRows scratch(rows, 128);
        const double* packed = scratch.pack(pi.data(), weights.data());
        assert(packed);
        for (int n : {0, 1, 2, 3, 7, 16, 31, 100}) {
            std::vector<int> indices(n + 3);
            std::vector<double> elements(n + 3);
            for (int trial = 0; trial < 1000; ++trial) {
                for (int j = 0; j < n + 3; ++j) {
                    indices[j] = random() % rows;
                    elements[j] = static_cast<int>(random() % 20001) / 8192.43;
                    if (j % 2)
                        elements[j] = -elements[j];
                }
                double expected = 0., expectedWeight = 0.;
                for (int j = 3; j < n + 3; ++j) {
                    expected -= pi[indices[j]] * elements[j];
                    expectedWeight += weights[indices[j]] * elements[j];
                }
                double value, weight;
                baldes_clp::transpose_pair(3, n + 3, indices.data(), elements.data(), packed, value,
                                           weight);
                assert(same(expected, value) && same(expectedWeight, weight));
            }
        }
    }
    for (double p : {0., -0., std::numeric_limits<double>::denorm_min(),
                     -std::numeric_limits<double>::denorm_min(), 1.e200, -1.e200}) {
        for (double coefficient : {0., -0., 1., -1., 1.e200, 1.e-200}) {
            double pi[] = {p}, weight[] = {-p};
            baldes_clp::PairedRows scratch(1, 128);
            const double* packed = scratch.pack(pi, weight);
            int row[] = {0, 0, 0};
            double elements[] = {coefficient, coefficient, coefficient};
            double expected = 0., expectedWeight = 0.;
            for (int j = 0; j < 3; ++j) {
                expected -= pi[0] * elements[j];
                expectedWeight += weight[0] * elements[j];
            }
            double value, actualWeight;
            baldes_clp::transpose_pair(0, 3, row, elements, packed, value, actualWeight);
            assert(same(expected, value) && same(expectedWeight, actualWeight));
        }
    }
    double x = 1.;
    baldes_clp::PairedRows tiny(1, 127);
    assert(tiny.pack(&x, &x) == nullptr);
    std::puts("paired sparse dot: 48,036 bitwise/NaN oracle cases passed");
#endif
}
