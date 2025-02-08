#include <execution>
#include <stdexec/execution.hpp>
#include <vector>

#include <execution>

// Helper function to wrap the bulk execution of a task
template <typename Func>
void bulk_parallel_for(size_t start, size_t end, Func &&func) {
    exec::static_thread_pool            pool  = exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    stdexec::bulk(stdexec::schedule(sched), end - start, [=](size_t idx) { func(idx + start); });
}

template <typename T>
inline T amd_flip(const T &i) {
    return -i - 2;
}

template <typename Scalar, typename StorageIndex>
inline void AMDord(Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>             &C,
                   Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, StorageIndex> &perm) {
    using std::sqrt;

    StorageIndex d, dk, dext, lemax = 0, e, elenk, eln, i, j, k, k1, k2, k3, jlast, ln, dense, nzmax, mindeg = 0, nvi,
                              nvj, nvk, mark, wnvi, ok, nel = 0, p, p1, p2, p3, p4, pj, pk, pk1, pk2, pn, q, t, h;

    StorageIndex n = StorageIndex(C.cols());
    dense          = std::max<StorageIndex>(16, StorageIndex(10 * sqrt(double(n)))); /* find dense threshold */
    dense          = (std::min)(n - 2, dense);

    StorageIndex cnz = StorageIndex(C.nonZeros());
    perm.resize(n + 1);
    t = cnz + cnz / 5 + 2 * n; /* add elbow room to C */
    C.resizeNonZeros(t);

    // get workspace
    std::vector<StorageIndex> W(8 * (n + 1), 0); // Equivalent to the old stack allocation

    std::vector<StorageIndex> len(n + 1, 0), nv(n + 1, 1), next(n + 1, -1), head(n + 1, -1);
    std::vector<StorageIndex> elen(n + 1, 0), degree(n + 1, 0), w(n + 1, 1), hhead(n + 1, -1);

    // StorageIndex *last = perm.indices().data(); /* use P as workspace for last */
    std::vector<StorageIndex> last(n + 1, -1);

    /* --- Initialize quotient graph ---------------------------------------- */
    StorageIndex *Cp = C.outerIndexPtr();
    StorageIndex *Ci = C.innerIndexPtr();
    for (k = 0; k < n; k++) len[k] = Cp[k + 1] - Cp[k];
    len[n] = 0;
    nzmax  = t;

    for (i = 0; i <= n; i++) {
        // head[i]   = -1; // degree list i is empty
        last[i] = -1;

        degree[i] = len[i]; // degree of node i
    }
    mark = Eigen::internal::cs_wclear<StorageIndex>(0, 0, w.data(), n);

    /* --- Initialize degree lists ------------------------------------------ */
    for (StorageIndex i = 0; i < n; i++) {
        bool has_diag = false;

        // Check for diagonal element in the row
        for (StorageIndex p = Cp[i]; p < Cp[i + 1]; ++p) {
            if (Ci[p] == i) {
                has_diag = true;
                break; // Exit early once diagonal is found
            }
        }

        StorageIndex d = degree[i];

        // If the node has degree 1 and a diagonal, mark it as dead
        if (d == 1 && has_diag) {
            elen[i] = -2; // Mark node i as dead
            nel++;        // Increment dead element count
            Cp[i] = -1;   // Mark i as a root of assembly tree
            w[i]  = 0;    // Reset weight
        }
        // If the node is dense or has no diagonal, absorb it into element n
        else if (d > dense || !has_diag) {
            nv[i]   = 0;         // Absorb node i into element n
            elen[i] = -1;        // Mark node i as dead
            nel++;               // Increment dead element count
            Cp[i] = amd_flip(n); // Mark absorbed into n
            nv[n]++;             // Increase the size of element n
        }
        // Otherwise, add node i to the degree list
        else {
            if (head[d] != -1) {
                last[head[d]] = i; // Link to previous node in degree list
            }
            next[i] = head[d]; // Add node i to degree list
            head[d] = i;       // Update degree list head
        }
    }

    elen[n] = -2; /* n is a dead element */
    Cp[n]   = -1; /* n is a root of assembly tree */
    w[n]    = 0;  /* n is a dead element */

    while (nel < n) /* while (selecting pivots) do */
    {
        /* --- Select node of minimum approximate degree -------------------- */
        StorageIndex k = -1;
        for (; mindeg < n && (k = head[mindeg]) == -1; ++mindeg) {}
        if (next[k] != -1) last[next[k]] = -1;
        head[mindeg] = next[k];

        StorageIndex elenk = elen[k], nvk = nv[k];
        nel += nvk;

        if (elenk > 0 && cnz + mindeg >= t) {
            for (StorageIndex j = 0; j < n; ++j) {
                if (StorageIndex p = Cp[j]; p >= 0) {
                    Cp[j] = Ci[p];
                    Ci[p] = amd_flip(j);
                }
            }

            for (StorageIndex q = 0, p = 0; p < cnz;) {
                if (StorageIndex j = amd_flip(Ci[p++]); j >= 0) {
                    Ci[q] = Cp[j];
                    Cp[j] = q++;
                    for (StorageIndex k3 = 0; k3 < len[j] - 1; ++k3) Ci[q++] = Ci[p++];
                }
            }
            cnz = q;
        }

        StorageIndex dk = 0;
        nv[k]           = -nvk;
        StorageIndex p = Cp[k], pk1 = (elenk == 0) ? p : cnz, pk2 = pk1;

        for (StorageIndex k1 = 1; k1 <= elenk + 1; ++k1) {
            StorageIndex e, pj, ln;
            if (k1 > elenk) {
                e  = k;
                pj = p;
                ln = len[k] - elenk;
            } else {
                e  = Ci[p++];
                pj = Cp[e];
                ln = len[e];
            }

            for (StorageIndex k2 = 1; k2 <= ln; ++k2) {
                StorageIndex i = Ci[pj++];
                if (nv[i] <= 0) continue;
                dk += nv[i];
                nv[i]     = -nv[i];
                Ci[pk2++] = i;
                if (next[i] != -1) last[next[i]] = last[i];
                if (last[i] != -1) {
                    next[last[i]] = next[i];
                } else {
                    head[degree[i]] = next[i];
                }
            }

            if (e != k) {
                Cp[e] = amd_flip(k);
                w[e]  = 0;
            }
        }

        if (elenk != 0) cnz = pk2; /* Ci[cnz...nzmax] is free */
        degree[k] = dk;            /* external degree of k - |Lk\i| */
        Cp[k]     = pk1;           /* element k is in Ci[pk1..pk2-1] */
        len[k]    = pk2 - pk1;
        elen[k]   = -2; /* k is now an element */

        /* --- Find set differences ----------------------------------------- */
        mark = Eigen::internal::cs_wclear<StorageIndex>(mark, lemax, w.data(), n); // Clear w if necessary
        for (StorageIndex pk = pk1; pk < pk2; ++pk) {
            StorageIndex i = Ci[pk];
            if (elen[i] <= 0) continue; // Skip if elen[i] is non-positive

            StorageIndex nvi   = -nv[i];
            StorageIndex wnvi  = mark - nvi;
            StorageIndex p_end = Cp[i] + elen[i]; // Precompute loop bound

            // Unroll this loop to reduce branching overhead
            for (StorageIndex p = Cp[i]; p < p_end; ++p) {
                StorageIndex e = Ci[p];
                if (w[e] >= mark) {
                    w[e] -= nvi;
                } else if (w[e] != 0) {
                    w[e] = degree[e] + wnvi;
                }
            }
        }

        /* --- Degree update ------------------------------------------------ */
        for (StorageIndex pk = pk1; pk < pk2; ++pk) {
            StorageIndex i  = Ci[pk];
            StorageIndex p1 = Cp[i];
            StorageIndex p2 = p1 + elen[i]; // Precompute loop bounds
            StorageIndex pn = p1;

            StorageIndex d = 0; // Degree accumulator
            StorageIndex h = 0; // Hash accumulator

            // This loop can potentially benefit from unrolling
            for (StorageIndex p = p1; p < p2; ++p) {
                StorageIndex e = Ci[p];
                if (w[e] != 0) {
                    StorageIndex dext = w[e] - mark;
                    if (dext > 0) {
                        d += dext;
                        Ci[pn++] = e;
                        h += e;
                    } else {
                        Cp[e] = amd_flip(k); // Absorb node e
                        w[e]  = 0;
                    }
                }
            }

            elen[i] = pn - p1 + 1;
            if (d == 0) { // Node i is now dead
                Cp[i]            = amd_flip(k);
                StorageIndex nvi = -nv[i];
                dk -= nvi;
                nvk += nvi;
                nel += nvi;
                nv[i]   = 0;
                elen[i] = -1;
            } else {
                degree[i] = std::min(degree[i], d); // Update degree
                Ci[pn]    = Ci[p1 + len[i]];        // Compact the list
                Ci[p1]    = k;
                len[i]    = pn - p1 + 1;

                // Hash insertion for degree list
                h %= n;
                next[i]  = hhead[h];
                hhead[h] = i;
                last[i]  = h;
            }
        }

        degree[k] = dk;

        lemax = std::max<StorageIndex>(lemax, dk);
        mark  = Eigen::internal::cs_wclear<StorageIndex>(mark + lemax, lemax, w.data(), n); /* clear w */

        /* --- Supernode detection ------------------------------------------ */
        for (StorageIndex pk = pk1; pk < pk2; pk++) {
            i = Ci[pk];
            if (nv[i] >= 0) continue; /* skip if i is dead */

            h        = last[i]; /* scan hash bucket of node i */
            i        = hhead[h];
            hhead[h] = -1; /* hash bucket will be empty */

            while (i != -1 && next[i] != -1) {
                ln                         = len[i];
                eln                        = elen[i];
                const StorageIndex endCp_i = Cp[i] + ln;

                // Use a local variable to store Cp[i] + 1 instead of recalculating in each iteration
                StorageIndex startCp_i = Cp[i] + 1;

                // Unroll the inner loop to update w[Ci[p]] = mark
                for (p = startCp_i; p <= endCp_i - 1; p++) { w[Ci[p]] = mark; }

                jlast = i;
                j     = next[i];

                // Iterate through all j, comparing i with j
                while (j != -1) {
                    ok                     = (len[j] == ln) && (elen[j] == eln);
                    StorageIndex startCp_j = Cp[j] + 1;
                    StorageIndex endCp_j   = Cp[j] + ln;

                    for (p = startCp_j; ok && p <= endCp_j - 1; p++) {
                        if (w[Ci[p]] != mark) {
                            ok = 0; // i and j are different
                        }
                    }

                    if (ok) {                // i and j are identical
                        Cp[j] = amd_flip(i); // absorb j into i
                        nv[i] += nv[j];
                        nv[j]       = 0;
                        elen[j]     = -1; // mark j as dead
                        j           = next[j];
                        next[jlast] = j; // delete j from hash bucket
                    } else {
                        jlast = j; // j and i are different
                        j     = next[j];
                    }
                }
                i = next[i]; // move to next in the hash bucket
                mark++;
            }
        }

        /* --- Supernode detection ------------------------------------------ */
        for (pk = pk1; pk < pk2; pk++) {
            i = Ci[pk];
            if (nv[i] >= 0) continue; /* skip if i is dead */

            h        = last[i]; /* scan hash bucket of node i */
            i        = hhead[h];
            hhead[h] = -1; /* hash bucket will be empty */

            while (i != -1 && next[i] != -1) {
                ln                         = len[i];
                eln                        = elen[i];
                const StorageIndex endCp_i = Cp[i] + ln;

                // Use a local variable to store Cp[i] + 1 instead of recalculating in each iteration
                StorageIndex startCp_i = Cp[i] + 1;

                // Unroll the inner loop to update w[Ci[p]] = mark
                for (p = startCp_i; p <= endCp_i - 1; p++) { w[Ci[p]] = mark; }

                jlast = i;
                j     = next[i];

                // Iterate through all j, comparing i with j
                while (j != -1) {
                    ok                     = (len[j] == ln) && (elen[j] == eln);
                    StorageIndex startCp_j = Cp[j] + 1;
                    StorageIndex endCp_j   = Cp[j] + ln;

                    for (p = startCp_j; ok && p <= endCp_j - 1; p++) {
                        if (w[Ci[p]] != mark) {
                            ok = 0; // i and j are different
                        }
                    }

                    if (ok) {                // i and j are identical
                        Cp[j] = amd_flip(i); // absorb j into i
                        nv[i] += nv[j];
                        nv[j]       = 0;
                        elen[j]     = -1; // mark j as dead
                        j           = next[j];
                        next[jlast] = j; // delete j from hash bucket
                    } else {
                        jlast = j; // j and i are different
                        j     = next[j];
                    }
                }
                i = next[i]; // move to next in the hash bucket
                mark++;
            }
        }

        /* --- Finalize new element ------------------------------------------ */
        for (p = pk1, pk = pk1; pk < pk2; pk++) {
            i = Ci[pk];
            if ((nvi = -nv[i]) <= 0) continue; // skip if i is dead

            nv[i] = nvi;                  // restore nv[i]
            d     = degree[i] + dk - nvi; // compute external degree(i)
            d     = std::min<StorageIndex>(d, n - nel - nvi);

            // Improve degree list updates with fewer memory accesses
            if (head[d] != -1) last[head[d]] = i;
            next[i] = head[d]; // put i back in degree list
            last[i] = -1;
            head[d] = i;
            mindeg  = std::min<StorageIndex>(mindeg, d); // update minimum degree

            Ci[p++] = i; // place i in Lk
        }

        nv[k] = nvk;                   // # nodes absorbed into k
        if ((len[k] = p - pk1) == 0) { // length of adj list of element k
            Cp[k] = -1;                // k is a root of the tree
            w[k]  = 0;                 // k is now a dead element
        }

        if (elenk != 0) {
            cnz = p; // free unused space in Lk
        }
    }

    /* --- Postordering ----------------------------------------------------- */
    for (StorageIndex i = 0; i < n; ++i) Cp[i] = amd_flip(Cp[i]);
    for (StorageIndex j = 0; j <= n; ++j) head[j] = -1;
    for (StorageIndex j = n; j >= 0; --j) {
        if (nv[j] > 0) continue;
        next[j]     = head[Cp[j]];
        head[Cp[j]] = j;
    }

    for (StorageIndex e = n; e >= 0; --e) {
        if (nv[e] <= 0) continue;
        if (Cp[e] != -1) {
            next[e]     = head[Cp[e]];
            head[Cp[e]] = e;
        }
    }

    for (StorageIndex k = 0, i = 0; i <= n; ++i) {
        if (Cp[i] == -1)
            k = Eigen::internal::cs_tdfs<StorageIndex>(i, k, head.data(), next.data(), perm.indices().data(), w.data());
    }

    perm.indices().conservativeResize(n);
}
