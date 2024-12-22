#pragma once

#include <ankerl/unordered_dense.h>
#include <mutex>
#include <xxhash.h>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <shared_mutex>

class RouteCache {
private:
    struct alignas(64) Entry {
        std::vector<std::vector<int>> value;

        explicit Entry(const std::vector<std::vector<int>> &v) : value(v) {}
        explicit Entry(std::vector<std::vector<int>> &&v) : value(std::move(v)) {}
    };

    struct Hasher {
        size_t operator()(const std::vector<int> &v) const noexcept {
            return XXH3_64bits(v.data(), v.size() * sizeof(int));
        }
    };

    using EntryPtr = std::shared_ptr<Entry>;
    using CacheMap = ankerl::unordered_dense::map<std::vector<int>, EntryPtr, Hasher>;

    CacheMap map;
    mutable std::shared_mutex map_mutex; // Mutex for thread-safe map operations

    static thread_local std::pair<std::vector<int>, EntryPtr> local_cache;

public:
    bool try_get(const std::vector<int> &key, std::vector<std::vector<int>> &out_value) {
        // Check thread-local cache first
        if (local_cache.first == key && local_cache.second) {
            out_value = local_cache.second->value;
            return true;
        }

        // Acquire shared lock for map lookup
        std::shared_lock lock(map_mutex);
        auto it = map.find(key);
        if (it == map.end() || !it->second) return false;

        out_value = it->second->value;
        local_cache = {key, it->second}; // Update thread-local cache
        return true;
    }

    void put(std::vector<int> key, const std::vector<std::vector<int>> &value) {
        auto new_entry = std::make_shared<Entry>(value);

        // Acquire unique lock for map modification
        std::unique_lock lock(map_mutex);
        auto [it, inserted] = map.emplace(std::move(key), new_entry);
        if (!inserted) {
            it->second = new_entry; // Update the existing entry
        }

        // Update thread-local cache
        local_cache = {it->first, new_entry};
    }

    static void invalidate_local_cache() noexcept {
        local_cache = {{}, nullptr};
    }

    void clear() {
        // Invalidate thread-local cache
        invalidate_local_cache();

        // Acquire unique lock and clear the map
        std::unique_lock lock(map_mutex);
        map.clear();
    }
};

