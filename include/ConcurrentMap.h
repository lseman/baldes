#pragma once
#include <array>
#include <atomic>
#include <functional>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

template <typename K, typename V, typename Hash = std::hash<K>>
class ConcurrentHashMap {
public:
    // Increased default bucket count for better initial distribution
    ConcurrentHashMap(size_t bucket_count = 64) {
        size_t per_shard = std::max(size_t(64), bucket_count / NUM_SHARDS);
        // Round up to next power of 2
        per_shard = nextPowerOf2(per_shard);
        for (auto& shard : shards) {
            shard = std::make_unique<Shard>(per_shard);
        }
    }

    // Move constructor
    ConcurrentHashMap(ConcurrentHashMap&& other) noexcept 
        : shards(std::move(other.shards)) {
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            shard_sizes[i].store(
                other.shard_sizes[i].load(std::memory_order_relaxed),
                std::memory_order_relaxed
            );
        }
    }

    // Move assignment operator
    ConcurrentHashMap& operator=(ConcurrentHashMap&& other) noexcept {
        if (this != &other) {
            shards = std::move(other.shards);
            for (size_t i = 0; i < NUM_SHARDS; ++i) {
                shard_sizes[i].store(
                    other.shard_sizes[i].load(std::memory_order_relaxed),
                    std::memory_order_relaxed
                );
            }
        }
        return *this;
    }

    bool insert(const K& key, V value) {  // Note: value passed by value for strong exception guarantee
        const size_t hashval = calculateHash(key);
        auto& shard = shards[getShardIndex(hashval)];
        const size_t bucket_idx = shard->getBucketIndex(hashval);
        auto& bucket = shard->buckets[bucket_idx];

        {
            // Try read-lock first to check existence
            std::shared_lock read_lock(bucket.mutex);
            if (bucket.items.find(key) != bucket.items.end()) {
                return false;  // Key exists
            }
        }

        // Key likely doesn't exist, acquire write lock
        std::unique_lock write_lock(bucket.mutex);
        
        // Double-check under write lock
        auto [it, inserted] = bucket.items.try_emplace(key, std::move(value));
        if (inserted) {
            shard_sizes[getShardIndex(hashval)].fetch_add(1, std::memory_order_relaxed);
        }
        return inserted;
    }

    std::optional<V> find(const K& key) const {
        const size_t hashval = calculateHash(key);
        const auto& shard = shards[getShardIndex(hashval)];
        const size_t bucket_idx = shard->getBucketIndex(hashval);
        const auto& bucket = shard->buckets[bucket_idx];

        std::shared_lock lock(bucket.mutex);
        auto it = bucket.items.find(key);
        if (it != bucket.items.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    bool erase(const K& key) {
        const size_t hashval = calculateHash(key);
        auto& shard = shards[getShardIndex(hashval)];
        const size_t bucket_idx = shard->getBucketIndex(hashval);
        auto& bucket = shard->buckets[bucket_idx];

        std::unique_lock lock(bucket.mutex);
        if (bucket.items.erase(key) > 0) {
            shard_sizes[getShardIndex(hashval)].fetch_sub(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    template<typename F>
    bool update(const K& key, F&& updater) {
        const size_t hashval = calculateHash(key);
        auto& shard = shards[getShardIndex(hashval)];
        const size_t bucket_idx = shard->getBucketIndex(hashval);
        auto& bucket = shard->buckets[bucket_idx];

        std::unique_lock lock(bucket.mutex);
        auto it = bucket.items.find(key);
        if (it != bucket.items.end()) {
            return updater(it->second);
        }
        return false;
    }

    size_t size() const {
        size_t total = 0;
        for (const auto& count : shard_sizes) {
            total += count.load(std::memory_order_relaxed);
        }
        return total;
    }

    size_t capacity() const {
        size_t total = 0;
        for (const auto& shard : shards) {
            total += shard->buckets.size();
        }
        return total;
    }

    void clear() {
        for (auto& shard : shards) {
            for (auto& bucket : shard->buckets) {
                std::unique_lock lock(bucket.mutex);
                bucket.items.clear();
            }
        }
        for (auto& count : shard_sizes) {
            count.store(0, std::memory_order_relaxed);
        }
    }

    // Thread-safe bucket statistics
    struct BucketStats {
        size_t total_buckets;
        size_t total_items;
        double average_load;
        size_t max_bucket_size;
        size_t min_bucket_size;
    };

    BucketStats getBucketStats() const {
        BucketStats stats{0, 0, 0.0, 0, std::numeric_limits<size_t>::max()};
        
        for (const auto& shard : shards) {
            for (const auto& bucket : shard->buckets) {
                std::shared_lock lock(bucket.mutex);
                const size_t bucket_size = bucket.items.size();
                stats.total_buckets++;
                stats.total_items += bucket_size;
                stats.max_bucket_size = std::max(stats.max_bucket_size, bucket_size);
                stats.min_bucket_size = std::min(stats.min_bucket_size, bucket_size);
            }
        }
        
        if (stats.total_buckets > 0) {
            stats.average_load = static_cast<double>(stats.total_items) / stats.total_buckets;
        }
        
        return stats;
    }

    // Optional: Get load factor threshold
    double getLoadFactor() const {
        return static_cast<double>(size()) / capacity();
    }

private:
    static constexpr size_t NUM_SHARDS = 32;  // Increased from 16
    static constexpr size_t SHARD_MASK = NUM_SHARDS - 1;
    static constexpr size_t HASH_BITS = sizeof(size_t) * 8;

    // Improved cache line alignment
    struct alignas(128) Bucket {  // Increased from 64
        std::unordered_map<K, V, Hash> items;
        mutable std::shared_mutex mutex;
        char padding[128 - sizeof(std::unordered_map<K,V,Hash>) - sizeof(std::shared_mutex)];
    };

    struct alignas(128) Shard {
        std::vector<Bucket> buckets;
        Hash hasher;
        
        explicit Shard(size_t n) : buckets(n) {}
        
        size_t getBucketIndex(size_t hashval) const {
            return hashval & (buckets.size() - 1);
        }
    };

    // Member variables
    std::array<std::unique_ptr<Shard>, NUM_SHARDS> shards;
    std::array<std::atomic<size_t>, NUM_SHARDS> shard_sizes{};

    // Improved hash calculation
    [[nodiscard]] inline size_t calculateHash(const K& key) const {
        return shards[0]->hasher(key);
    }

    [[nodiscard]] inline size_t getShardIndex(size_t hashval) const {
        // Better distribution using upper bits
        return (hashval >> (HASH_BITS - 5)) & SHARD_MASK;  // Using 5 bits for 32 shards
    }

    // Utility function to get next power of 2
    static size_t nextPowerOf2(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }
};