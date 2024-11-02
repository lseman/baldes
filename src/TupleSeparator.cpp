
#include "TupleSeparator.h"

using RowPair = std::pair<int, int>;
using Rowset  = std::set<int>;

std::vector<Rowset> TupleBasedSeparator::separate4R1Cs() {
    std::vector<Rowset>      violatedCuts;
    std::mutex               cuts_mutex;
    const int                num_chunks = std::thread::hardware_concurrency();
    exec::static_thread_pool pool(std::thread::hardware_concurrency());
    auto                     sched = pool.get_scheduler();
    const int                chunk_size = rowPairs.size() / std::thread::hardware_concurrency();

    // Define parallel task for each chunk
    auto chunk_sender = stdexec::bulk(
        stdexec::just(), num_chunks, [this, &cuts_mutex, chunk_size, &violatedCuts](std::size_t chunk_idx) {
            size_t start_idx = chunk_idx * chunk_size;
            size_t end_idx   = std::min(start_idx + chunk_size, rowPairs.size());

            for (size_t i = start_idx; i < end_idx; ++i) {
                for (size_t j = i + 1; j < rowPairs.size(); ++j) {
                    Rowset C = combineRowPairs(rowPairs[i], rowPairs[j]);
                    // print C
                    //fmt::print("C: ");
                    //for (auto row : C) { fmt::print("{} ", row); }

                    if (C.size() == 4 && exploredRowsets.find(C) == exploredRowsets.end()) {
                        auto omega2 = getOmega2(C);

                        for (const auto &multipliers : multiplierSets) {
                        //fmt::print("Multipliers: ");
                            double violation = computeViolation(C, multipliers, omega2);
                            if (violation > violation_tolerance) {
                               // fmt::print("Violation: {}\n", violation);
                                std::lock_guard<std::mutex> lock(cuts_mutex);
                                violatedCuts.push_back(C);
                                break;
                            }
                        }
                        std::lock_guard<std::mutex> lock(cuts_mutex);
                        exploredRowsets.insert(C);
                    }
                }
            }
        });

    // Launch and synchronize the bulk sender
    auto work = stdexec::starts_on(sched, chunk_sender);
    stdexec::sync_wait(std::move(work));

    return removeDuplicateCuts(violatedCuts);
}