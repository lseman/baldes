/*
 * @file ArcNGMemory.h
 * @brief Physical-arc NG memories with grouped elementarity sets, two-phase
 *        cycle strengthening, and legacy DSSR repricing path.
 *
 * Ported from openrscp/pricing/ArcNGMemory.h. Value-owned global set IDs
 * replace head-local packed coordinates.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>



class ArcNGMemory {
public:
    struct Arc { int id, from, to; std::vector<int> visited_sets; };
    struct Column { std::vector<int> arc_ids; double value = 0; };
    struct Controls {
        int mode = 1;
        int max_vertex_size = 0;
        int max_average_size = 0;
        int max_columns = 100;
        int max_cycle_size = 5;
    };
    struct Result {
        std::size_t non_elementary = 0;
        std::size_t elementary = 0;
        std::size_t columns_forbidden = 0;
        std::size_t cycles_strengthened = 0;
        std::size_t phase_one_columns = 0;
        std::size_t phase_two_columns = 0;
        bool changed = false;
    };

    bool enabled = false;
    int set_count = 0;
    std::vector<std::vector<int>> vertex_sets;
    std::vector<std::vector<int>> vertex_memory;
    std::map<int, Arc> arcs;
    std::vector<int> arc_order;
    std::map<int, std::vector<int>> arc_memory;
    std::vector<std::vector<int>> initial_vertex_memory;
    std::map<int, std::vector<int>> initial_arc_memory;

    // Legacy DSSR preparation: restores initial vertex and
    // physical-arc memories before the reduction pricing loop.
    void save_initial_memories() {
        if (!enabled) throw std::logic_error("initialize NG memories first");
        initial_vertex_memory = vertex_memory;
        initial_arc_memory = arc_memory;
    }
    bool restore_initial_memories() {
        if (!enabled) throw std::logic_error("initialize NG memories first");
        const bool changed = vertex_memory != initial_vertex_memory ||
                             arc_memory != initial_arc_memory;
        vertex_memory = initial_vertex_memory;
        arc_memory = initial_arc_memory;
        return changed;
    }

    static void normalize(std::vector<int>& sets) {
        std::sort(sets.begin(), sets.end());
        sets.erase(std::unique(sets.begin(), sets.end()), sets.end());
    }
    static std::vector<int> intersection(const std::vector<int>& a,
                                         const std::vector<int>& b) {
        std::vector<int> result;
        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                              std::back_inserter(result));
        return result;
    }
    static void merge(std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> result;
        std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                       std::back_inserter(result));
        a = std::move(result);
    }
    static bool subset(const std::vector<int>& a, const std::vector<int>& b) {
        return std::includes(b.begin(), b.end(), a.begin(), a.end());
    }

    void initialize(int groups, std::vector<std::vector<int>> membership,
                    std::vector<std::vector<int>> neighborhoods,
                    const std::vector<Arc>& physical_arcs) {
        if (groups < 0 || groups > 1024 || membership.size() != neighborhoods.size())
            throw std::invalid_argument("NG elementarity set dimensions");
        const auto check = [&](std::vector<int>& ids) {
            normalize(ids);
            for (int id : ids)
                if (id < 0 || id >= groups)
                    throw std::out_of_range("NG elementarity set");
        };
        for (auto& ids : membership) check(ids);
        for (auto& ids : neighborhoods) check(ids);
        std::map<int, Arc> next_arcs;
        std::vector<int> next_order;
        for (auto arc : physical_arcs) {
            if (arc.id < 0 || arc.from < 0 || arc.to < 0 ||
                static_cast<std::size_t>(arc.from) >= membership.size() ||
                static_cast<std::size_t>(arc.to) >= membership.size())
                throw std::invalid_argument("NG physical arc");
            check(arc.visited_sets);
            next_order.push_back(arc.id);
            if (!next_arcs.emplace(arc.id, std::move(arc)).second)
                throw std::invalid_argument("duplicate NG physical arc");
        }
        set_count = groups;
        vertex_sets = std::move(membership);
        vertex_memory = std::move(neighborhoods);
        arcs = std::move(next_arcs);
        arc_order = std::move(next_order);
        arc_memory.clear();
        enabled = true;
        save_initial_memories();
    }

    //: distances are indexed by elementarity set, not vertex.
    // Native traversal stops collecting destinations at the first nonempty
    // memory independently for vertices and physical arcs.
    bool initialize_from_set_distances(const std::vector<std::vector<double>>& distance,
                                       std::size_t size) {
        if (!enabled) throw std::logic_error("initialize NG memories first");
        if (distance.empty()) return false;
        const auto n = static_cast<std::size_t>(set_count);
        if (distance.size() != n)
            throw std::invalid_argument("NG set distance matrix size");
        for (const auto& row : distance) {
            if (row.size() != n)
                throw std::invalid_argument("NG set distance matrix size");
            for (double value : row)
                if (std::isnan(value))
                    throw std::invalid_argument("NG set distance is NaN");
        }
        std::vector<std::vector<int>> neighborhoods(n);
        for (std::size_t set = 0; set < n; ++set) {
            std::vector<std::pair<double, int>> order;
            for (std::size_t other = 0; other < n; ++other)
                order.emplace_back(distance[set][other], static_cast<int>(other));
            std::stable_sort(order.begin(), order.end());
            for (std::size_t i = 0; i < std::min(n, size); ++i)
                neighborhoods[set].push_back(order[i].second);
            normalize(neighborhoods[set]);
        }
        bool changed = false;
        for (std::size_t v = 0; v < vertex_memory.size(); ++v) {
            auto& memory = vertex_memory[v];
            if (!memory.empty()) break;
            for (int set : vertex_sets[v])
                merge(memory, neighborhoods[static_cast<std::size_t>(set)]);
            changed = changed || !memory.empty();
            initial_vertex_memory[v] = memory;
        }
        for (int id : arc_order) {
            auto& memory = arc_memory[id];
            if (!memory.empty()) break;
            for (int set : arcs.at(id).visited_sets)
                merge(memory, neighborhoods[static_cast<std::size_t>(set)]);
            changed = changed || !memory.empty();
            initial_arc_memory[id] = memory;
        }
        return changed;
    }

    std::vector<int> kept_sets(const Arc& arc) const {
        auto tail = vertex_memory.at(static_cast<std::size_t>(arc.from));
        auto head = vertex_memory.at(static_cast<std::size_t>(arc.to));
        merge(tail, vertex_sets[static_cast<std::size_t>(arc.from)]);
        merge(head, vertex_sets[static_cast<std::size_t>(arc.to)]);
        auto kept = intersection(tail, head);
        const auto extra = arc_memory.find(arc.id);
        if (extra != arc_memory.end()) merge(kept, extra->second);
        return kept;
    }
    std::vector<int> visited_sets(const Arc& arc) const {
        auto visited = vertex_sets.at(static_cast<std::size_t>(arc.to));
        merge(visited, arc.visited_sets);
        return visited;
    }
    bool extend(std::vector<int>& next, const std::vector<int>& state,
                int original_arc) const {
        const auto& arc = arcs.at(original_arc);
        const auto visited = visited_sets(arc);
        if (!intersection(state, visited).empty()) return false;
        next = intersection(state, kept_sets(arc));
        merge(next, visited);
        return true;
    }
    // Symbolically replay the suffix: unknown prefix memories survive only
    // remembered arcs; each encountered set contributes to the prefix-forbid set.
    bool suffix_forbidden(int vertex, const std::vector<int>& path,
                          std::vector<int>& forbidden) const {
        auto known = vertex_sets.at(static_cast<std::size_t>(vertex));
        std::vector<int> unknown(static_cast<std::size_t>(set_count));
        std::iota(unknown.begin(), unknown.end(), 0);
        forbidden.clear();
        for (int id : path) {
            const auto& arc = arcs.at(id);
            if (arc.from != vertex) return false;
            const auto visited = visited_sets(arc);
            if (!intersection(known, visited).empty()) return false;
            merge(forbidden, intersection(unknown, visited));
            const auto kept = kept_sets(arc);
            unknown = intersection(unknown, kept);
            known = intersection(known, kept);
            merge(known, visited);
            vertex = arc.to;
        }
        return true;
    }
    bool path_feasible(const std::vector<int>& path) const {
        if (path.empty()) return true;
        int vertex = arcs.at(path.front()).from;
        auto state = vertex_sets.at(static_cast<std::size_t>(vertex));
        for (int id : path) {
            const auto& arc = arcs.at(id);
            if (arc.from != vertex) return false;
            std::vector<int> next;
            if (!extend(next, state, id)) return false;
            state = std::move(next);
            vertex = arc.to;
        }
        return true;
    }

    // Legacy NG memory increase and cycle forbidding.
    // Reduction may only remember initial permitted sets. Augmentation may
    // enlarge that permitted domain, subject to an atomic per-cycle cap.
    Result strengthen_legacy(const std::vector<Column>& columns,
                             const Controls& controls, bool reduction,
                             const std::map<int, int>& opposite = {}) {
        if (!enabled) throw std::logic_error("initialize NG memories first");
        Result result;
        struct Cycle {
            std::size_t begin, end;
            int set;
            bool covered = false;
        };
        const bool vertex_mode = controls.mode == 1;
        for (const auto& column : columns) {
            const auto& path = column.arc_ids;
            std::vector<Cycle> cycles;
            std::map<int, std::size_t> last;
            if (!path.empty())
                for (int set :
                     vertex_sets.at(static_cast<std::size_t>(
                         arcs.at(path.front()).from)))
                    last[set] = 0;
            int vertex = path.empty() ? -1 : arcs.at(path.front()).from;
            for (std::size_t i = 0; i < path.size(); ++i) {
                const auto& arc = arcs.at(path[i]);
                if (arc.from != vertex)
                    throw std::invalid_argument(
                        "NG column is not a contiguous path");
                vertex = arc.to;
                for (int set : visited_sets(arc)) {
                    const auto found = last.find(set);
                    if (found != last.end())
                        cycles.push_back({found->second, i + 1, set, false});
                    last[set] = i + 1;
                }
            }
            if (cycles.empty()) {
                ++result.elementary;
                continue;
            }
            ++result.non_elementary;
            if (controls.max_columns <= 0 ||
                result.columns_forbidden >=
                    static_cast<std::size_t>(controls.max_columns))
                continue;
            std::stable_sort(
                cycles.begin(), cycles.end(),
                [](const Cycle& a, const Cycle& b) {
                    return a.end - a.begin < b.end - b.begin;
                });
            bool selected = false;
            for (int phase = 1; phase <= 2 && !selected; ++phase) {
                for (std::size_t c = 0; c < cycles.size(); ++c) {
                    auto& cycle = cycles[c];
                    if (phase == 1 &&
                        static_cast<long long>(cycle.end - cycle.begin) >=
                            controls.max_cycle_size)
                        break;
                    for (std::size_t previous = 0; previous < c; ++previous)
                        if (cycles[previous].covered &&
                            cycle.begin <= cycles[previous].begin &&
                            cycles[previous].end <= cycle.end)
                            cycle.covered = true;
                    if (cycle.covered) continue;
                    std::set<int> vertices;
                    for (std::size_t i = cycle.begin + 1; i < cycle.end; ++i)
                        vertices.insert(arcs.at(path[i]).from);
                    const int first = arcs.at(path[cycle.begin]).from;
                    const int final_ = arcs.at(path[cycle.end - 1]).to;
                    for (int endpoint : {first, final_})
                        if (std::binary_search(
                                vertex_sets[static_cast<std::size_t>(endpoint)]
                                    .begin(),
                                vertex_sets[static_cast<std::size_t>(endpoint)]
                                    .end(),
                                cycle.set))
                            vertices.insert(endpoint);
                    bool eligible = true;
                    for (int v : vertices) {
                        const auto& domain = reduction
                                                 ? initial_vertex_memory[static_cast<std::size_t>(v)]
                                                 : vertex_memory[static_cast<std::size_t>(v)];
                        const bool present =
                            std::binary_search(domain.begin(), domain.end(),
                                               cycle.set);
                        if ((reduction && !present) ||
                            (!reduction && !present &&
                             static_cast<long long>(domain.size()) >=
                                 controls.max_vertex_size)) {
                            eligible = false;
                            break;
                        }
                    }
                    if (!eligible) continue;
                    bool changed = false;
                    const auto add = [&](std::vector<int>& memory) {
                        if (std::binary_search(memory.begin(), memory.end(),
                                               cycle.set))
                            return;
                        memory.push_back(cycle.set);
                        normalize(memory);
                        changed = true;
                    };
                    if (vertex_mode)
                        for (int v : vertices)
                            add(vertex_memory[static_cast<std::size_t>(v)]);
                    else
                        for (std::size_t i = cycle.begin; i < cycle.end; ++i) {
                            add(arc_memory[path[i]]);
                            const auto mirror = opposite.find(path[i]);
                            if (mirror != opposite.end())
                                add(arc_memory[mirror->second]);
                        }
                    cycle.covered = true;
                    selected = true;
                    result.changed = result.changed || changed;
                    ++result.cycles_strengthened;
                    if (phase == 2) break;
                }
                if (selected) {
                    ++result.columns_forbidden;
                    if (phase == 1) ++result.phase_one_columns;
                    else ++result.phase_two_columns;
                }
            }
        }
        return result;
    }

    // Relaxation strengthening: descending fractional value,
    // ascending cycle size, strict phase-I cycle limit and vertex growth caps;
    // other modes strengthen physical arc memories. Symmetry mirrors changes.
    Result strengthen(const std::vector<Column>& columns,
                      const Controls& controls,
                      const std::map<int, int>& opposite = {}) {
        if (!enabled) throw std::logic_error("initialize NG memories first");
        Result result;
        const long long signed_mode = controls.mode;
        const long long mode = signed_mode < 0 ? -signed_mode : signed_mode;
        if (mode == 1) {
            std::size_t total = 0;
            bool all_at_cap = true;
            for (const auto& memory : vertex_memory) {
                total += memory.size();
                if (static_cast<long long>(memory.size()) <
                    controls.max_vertex_size)
                    all_at_cap = false;
            }
            if (all_at_cap ||
                (!vertex_memory.empty() &&
                 static_cast<long long>(total / vertex_memory.size()) >=
                     controls.max_average_size))
                return result;
        }
        std::vector<std::size_t> order(columns.size());
        std::iota(order.begin(), order.end(), 0);
        for (const auto& column : columns) {
            if (!std::isfinite(column.value))
                throw std::invalid_argument("NG column value");
            int vertex = column.arc_ids.empty() ? -1
                                                : arcs.at(column.arc_ids.front())
                                                      .from;
            for (int id : column.arc_ids) {
                const auto& arc = arcs.at(id);
                if (arc.from != vertex)
                    throw std::invalid_argument(
                        "NG column is not a contiguous path");
                vertex = arc.to;
            }
        }
        std::stable_sort(
            order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) {
                return columns[a].value > columns[b].value;
            });
        struct Cycle {
            std::size_t begin, end;
            int set;
        };
        for (std::size_t index : order) {
            const auto& path = columns[index].arc_ids;
            std::vector<Cycle> cycles;
            std::map<int, std::size_t> last;
            if (!path.empty())
                for (int set :
                     vertex_sets[static_cast<std::size_t>(
                         arcs.at(path.front()).from)])
                    last[set] = 0;
            for (std::size_t position = 0; position < path.size(); ++position)
                for (int set : visited_sets(arcs.at(path[position]))) {
                    const auto previous = last.find(set);
                    if (previous != last.end())
                        cycles.push_back(
                            {previous->second, position + 1, set});
                    last[set] = position + 1;
                }
            if (cycles.empty()) {
                ++result.elementary;
                continue;
            }
            ++result.non_elementary;
            if (controls.max_columns <= 0 ||
                result.columns_forbidden >=
                    static_cast<std::size_t>(controls.max_columns) ||
                !path_feasible(path))
                continue;
            std::stable_sort(
                cycles.begin(), cycles.end(),
                [](const Cycle& a, const Cycle& b) {
                    return a.end - a.begin < b.end - b.begin;
                });
            std::vector<Cycle> selected;
            int selected_phase = 0;
            for (int phase = 1; phase <= 2 && selected.empty(); ++phase)
                for (const auto& cycle : cycles) {
                    if (phase == 1 &&
                        static_cast<long long>(cycle.end - cycle.begin) >=
                            controls.max_cycle_size)
                        break;
                    if (std::any_of(selected.begin(), selected.end(),
                                    [&](const Cycle& old) {
                                        return cycle.begin <= old.begin &&
                                               old.end <= cycle.end;
                                    }))
                        continue;
                    bool changed = false;
                    for (std::size_t position = cycle.begin;
                         position < cycle.end; ++position) {
                        const auto& arc = arcs.at(path[position]);
                        if (mode == 1) {
                            auto& memory =
                                vertex_memory[static_cast<std::size_t>(
                                    arc.to)];
                            if (static_cast<long long>(memory.size()) >=
                                controls.max_vertex_size)
                                continue;
                            if (!std::binary_search(
                                    memory.begin(), memory.end(),
                                    cycle.set)) {
                                memory.push_back(cycle.set);
                                normalize(memory);
                                changed = true;
                            }
                        } else {
                            const auto add = [&](int id) {
                                auto& memory = arc_memory[id];
                                if (std::binary_search(memory.begin(),
                                                       memory.end(),
                                                       cycle.set))
                                    return false;
                                memory.push_back(cycle.set);
                                normalize(memory);
                                changed = true;
                                return true;
                            };
                            bool ok = add(arc_memory.find(path[position])
                                              != arc_memory.end()
                                              ? path[position]
                                              : -1);
                            if (!ok) continue;
                            const auto mirror = opposite.find(path[position]);
                            if (mirror != opposite.end()) {
                                if (!add(mirror->second)) {
                                    // Undo: remove from primary
                                    auto& mem = arc_memory[path[position]];
                                    mem.erase(
                                        std::remove(mem.begin(), mem.end(),
                                                    cycle.set),
                                        mem.end());
                                    normalize(mem);
                                    continue;
                                }
                            }
                        }
                    }
                    if (changed) {
                        selected.push_back(cycle);
                        selected_phase = phase;
                        result.changed = true;
                        ++result.cycles_strengthened;
                    }
                }
            if (!selected.empty()) {
                result.columns_forbidden += selected.size();
                result.phase_one_columns +=
                    selected_phase == 1 ? selected.size() : 0;
                result.phase_two_columns +=
                    selected_phase == 2 ? selected.size() : 0;
            }
        }
        return result;
    }
};


