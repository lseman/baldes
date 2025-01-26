#include <execution>
#include <stdexec/execution.hpp>
#include <vector>

// Helper function to wrap the bulk execution of a task
template <typename Func>
void bulk_parallel_for(size_t start, size_t end, Func&& func) {
    exec::static_thread_pool pool =
        exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    stdexec::bulk(stdexec::schedule(sched), end - start,
                  [=](size_t idx) { func(idx + start); });
}

template <typename T>
inline T amd_flip(const T& i) {
    return -i - 2;
}

template <typename Scalar, typename StorageIndex>
void AMDOrdering(Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>& C,
                 Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic,
                                          StorageIndex>& perm) {
    const StorageIndex n = C.cols();
    const StorageIndex dense = std::min(
        n - 2,
        std::max<StorageIndex>(16, StorageIndex(10 * std::sqrt(double(n)))));
    const StorageIndex cnz = C.nonZeros();
    const StorageIndex new_size = cnz + cnz / 5 + 2 * n;

    perm.resize(n + 1);
    C.resizeNonZeros(new_size);
    StorageIndex* const Cp = C.outerIndexPtr();
    StorageIndex* const Ci = C.innerIndexPtr();

    auto flip = [](StorageIndex i) { return -i - 2; };
    auto unflip = [](StorageIndex i) { return i < 0 ? -i - 2 : i; };

    std::vector<StorageIndex> len(n + 1), nv(n + 1, 1), next(n + 1, -1),
        head(n + 1, -1);
    std::vector<StorageIndex> elen(n + 1), degree(n + 1), w(n + 1, 1),
        hhead(n + 1, -1);
    std::vector<StorageIndex> last(n + 1, -1);

    // Initialize degree lists
    std::ranges::transform(std::views::iota(0, n), len.begin(),
                           [&](StorageIndex i) { return Cp[i + 1] - Cp[i]; });
    len[n] = 0;
    degree = len;

    StorageIndex nel = 0;
    StorageIndex mindeg = 0;
    StorageIndex mark = 0;

    // Process nodes
    for (StorageIndex i = 0; i < n; ++i) {
        bool has_diag =
            std::ranges::any_of(std::span(Ci + Cp[i], len[i]),
                                [i](StorageIndex idx) { return idx == i; });

        if (degree[i] == 1 && has_diag) {
            elen[i] = -2;
            nel++;
            Cp[i] = -1;
            w[i] = 0;
        } else if (degree[i] > dense || !has_diag) {
            nv[i] = 0;
            elen[i] = -1;
            nel++;
            Cp[i] = flip(n);
            nv[n]++;
        } else {
            if (head[degree[i]] != -1) {
                last[head[degree[i]]] = i;
            }
            next[i] = head[degree[i]];
            head[degree[i]] = i;
        }
    }

    elen[n] = -2;
    Cp[n] = -1;
    w[n] = 0;

    // Main elimination loop
    while (nel < n) {
        // Find minimum degree node
        StorageIndex k;
        for (; mindeg < n && (k = head[mindeg]) == -1; ++mindeg) {
        }

        if (next[k] != -1) last[next[k]] = -1;
        head[mindeg] = next[k];

        const StorageIndex elenk = elen[k];
        StorageIndex nvk = nv[k];
        nel += nvk;

        // Matrix compression if needed
        if (elenk > 0 && cnz + mindeg >= new_size) {
            for (StorageIndex j = 0; j < n; ++j) {
                if (StorageIndex p = Cp[j]; p >= 0) {
                    Cp[j] = Ci[p];
                    Ci[p] = flip(j);
                }
            }

            StorageIndex q = 0;
            for (StorageIndex p = 0; p < cnz;) {
                if (StorageIndex j = flip(Ci[p++]); j >= 0) {
                    Ci[q] = Cp[j];
                    Cp[j] = q++;
                    for (StorageIndex k3 = 0; k3 < len[j] - 1; ++k3) {
                        Ci[q++] = Ci[p++];
                    }
                }
            }
        }

        // Quotient graph update
        StorageIndex dk = 0;
        nv[k] = -nvk;
        StorageIndex p = Cp[k];
        StorageIndex pk1 = (elenk == 0) ? p : cnz;
        StorageIndex pk2 = pk1;

        for (StorageIndex k1 = 1; k1 <= elenk + 1; ++k1) {
            if (k1 > elenk) {
                StorageIndex e = k;
                StorageIndex pj = p;
                StorageIndex ln = len[k] - elenk;
                for (StorageIndex k2 = 1; k2 <= ln; ++k2) {
                    StorageIndex i = Ci[pj++];
                    if (nv[i] <= 0) continue;
                    dk += nv[i];
                    nv[i] = -nv[i];
                    Ci[pk2++] = i;
                    if (next[i] != -1) last[next[i]] = last[i];
                    if (last[i] != -1)
                        next[last[i]] = next[i];
                    else
                        head[degree[i]] = next[i];
                }
            } else {
                StorageIndex e = Ci[p++];
                StorageIndex pj = Cp[e];
                StorageIndex ln = len[e];
                for (StorageIndex k2 = 1; k2 <= ln; ++k2) {
                    StorageIndex i = Ci[pj++];
                    if (nv[i] <= 0) continue;
                    dk += nv[i];
                    nv[i] = -nv[i];
                    Ci[pk2++] = i;
                    if (next[i] != -1) last[next[i]] = last[i];
                    if (last[i] != -1)
                        next[last[i]] = next[i];
                    else
                        head[degree[i]] = next[i];
                }
                Cp[e] = flip(k);
                w[e] = 0;
            }
        }

        if (elenk != 0) cnz = pk2;
        degree[k] = dk;
        Cp[k] = pk1;
        len[k] = pk2 - pk1;
        elen[k] = -2;

        // Find set differences and update degrees
        mark++;
        for (StorageIndex pk = pk1; pk < pk2; ++pk) {
            StorageIndex i = Ci[pk];
            if (elen[i] <= 0) continue;

            StorageIndex nvi = -nv[i];
            StorageIndex wnvi = mark - nvi;

            for (StorageIndex p = Cp[i]; p < Cp[i] + elen[i]; ++p) {
                StorageIndex e = Ci[p];
                if (w[e] >= mark)
                    w[e] -= nvi;
                else if (w[e] != 0)
                    w[e] = degree[e] + wnvi;
            }
        }

        // Update element degrees
        for (StorageIndex pk = pk1; pk < pk2; ++pk) {
            StorageIndex i = Ci[pk];
            StorageIndex p1 = Cp[i];
            StorageIndex p2 = p1 + elen[i];
            StorageIndex pn = p1;
            StorageIndex d = 0;
            StorageIndex h = 0;

            for (StorageIndex p = p1; p < p2; ++p) {
                StorageIndex e = Ci[p];
                if (w[e] != 0) {
                    StorageIndex dext = w[e] - mark;
                    if (dext > 0) {
                        d += dext;
                        Ci[pn++] = e;
                        h += e;
                    } else {
                        Cp[e] = flip(k);
                        w[e] = 0;
                    }
                }
            }

            elen[i] = pn - p1 + 1;
            if (d == 0) {
                Cp[i] = flip(k);
                StorageIndex nvi = -nv[i];
                dk -= nvi;
                nvk += nvi;
                nel += nvi;
                nv[i] = 0;
                elen[i] = -1;
            } else {
                degree[i] = std::min(degree[i], d);
                Ci[pn] = Ci[p1 + len[i]];
                Ci[p1] = k;
                len[i] = pn - p1 + 1;
                h %= n;
                next[i] = hhead[h];
                hhead[h] = i;
                last[i] = h;
            }
        }

        degree[k] = dk;
        mark = std::max(mark + dk, mark);
        for (StorageIndex pk = pk1; pk < pk2; ++pk) {
            StorageIndex i = Ci[pk];
            if (nv[i] < 0) continue;
            StorageIndex p1 = Cp[i];
            StorageIndex p2 = p1 + elen[i] - 1;
            for (StorageIndex p = p1; p <= p2; ++p) {
                w[Ci[p]] = mark;
            }
        }

        // Finalize new element
        for (StorageIndex p = pk1, pk = pk1; pk < pk2; ++pk) {
            StorageIndex i = Ci[pk];
            StorageIndex nvi = -nv[i];
            if (nvi <= 0) continue;

            nv[i] = nvi;
            StorageIndex d = degree[i] + dk - nvi;
            d = std::min(d, n - nel - nvi);
            if (head[d] != -1) last[head[d]] = i;
            next[i] = head[d];
            last[i] = -1;
            head[d] = i;
            mindeg = std::min(mindeg, d);
            Ci[p++] = i;
        }

        nv[k] = nvk;
        if ((len[k] = pk2 - pk1) == 0) {
            Cp[k] = -1;
            w[k] = 0;
        }
        if (elenk != 0) cnz = pk2;
    }

    // Postordering
    std::ranges::transform(std::views::iota(0, n), Cp, Cp, flip);
    std::fill(head.begin(), head.end(), -1);

    for (StorageIndex j = n; j >= 0; --j) {
        if (nv[j] > 0) continue;
        next[j] = head[Cp[j]];
        head[Cp[j]] = j;
    }

    for (StorageIndex e = n; e >= 0; --e) {
        if (nv[e] <= 0) continue;
        if (Cp[e] != -1) {
            next[e] = head[Cp[e]];
            head[Cp[e]] = e;
        }
    }

    StorageIndex k = 0;
    for (StorageIndex i = 0; i <= n; ++i) {
        if (Cp[i] == -1) {
            for (StorageIndex j = i; j != -1; j = next[j]) {
                if (w[j] != 1) continue;
                w[j] = 1;
                perm.indices()[k++] = j;
            }
        }
    }

    perm.indices().conservativeResize(n);
}
