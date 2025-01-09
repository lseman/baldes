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
    // Increased default bucket count and shards for better distribution
    static constexpr size_t NUM_SHARDS = 64;  // Doubled from 32
    
    ConcurrentHashMap(size_t bucket_count = 128) {
        size_t per_shard = std::max(size_t(128), bucket_count / NUM_SHARDS);
        per_shard = nextPowerOf2(per_shard);
        for (auto& shard : shards) {
            shard = std::make_unique<Shard>(per_shard);
        }
    }

    bool insert(const K& key, V value) {
        const size_t hashval = calculateHash(key);
        auto& shard = shards[getShardIndex(hashval)];
        const size_t bucket_idx = shard->getBucketIndex(hashval);
        auto& bucket = shard->buckets[bucket_idx];

        // Use try_lock for better performance under contention
        if (bucket.mutex.try_lock_shared()) {
            if (bucket.items.find(key) != bucket.items.end()) {
                bucket.mutex.unlock_shared();
                return false;
            }
            bucket.mutex.unlock_shared();
        }

        // Optimistic locking - try exclusive lock directly
        std::unique_lock write_lock(bucket.mutex);
        auto [it, inserted] = bucket.items.try_emplace(key, std::move(value));
        if (inserted) {
            shard_sizes[getShardIndex(hashval)].fetch_add(1, std::memory_order_relaxed);
            
            // Dynamic resizing check
            if (bucket.items.size() > BUCKET_RESIZE_THRESHOLD) {
                triggerResize(shard.get(), bucket_idx);
            }
        }
        return inserted;
    }

    std::optional<V> find(const K& key) const {
        const size_t hashval = calculateHash(key);
        const auto& shard = shards[getShardIndex(hashval)];
        const size_t bucket_idx = shard->getBucketIndex(hashval);
        const auto& bucket = shard->buckets[bucket_idx];

        // Try lock optimization
        if (!bucket.mutex.try_lock_shared()) {
            return std::nullopt;  // Return early under high contention
        }
        
        auto it = bucket.items.find(key);
        if (it != bucket.items.end()) {
            V result = it->second;  // Make a copy while under lock
            bucket.mutex.unlock_shared();
            return result;
        }
        bucket.mutex.unlock_shared();
        return std::nullopt;
    }

private:
    static constexpr size_t SHARD_MASK = NUM_SHARDS - 1;
    static constexpr size_t HASH_BITS = sizeof(size_t) * 8;
    static constexpr size_t BUCKET_RESIZE_THRESHOLD = 32;  // Threshold for bucket splitting

    // Improved bucket structure with better memory layout
    struct alignas(256) Bucket {  // Increased alignment for better cache behavior
        std::unordered_map<K, V, Hash> items;
        mutable std::shared_mutex mutex;
        
        // Pre-allocated memory pool for small insertions
        static constexpr size_t POOL_SIZE = 8;
        std::array<std::pair<K, V>, POOL_SIZE> small_pool;
        size_t pool_size{0};
        
        // Default constructor
        Bucket() = default;
        
        // Move constructor
        Bucket(Bucket&& other) noexcept 
            : items(std::move(other.items))
            , small_pool(std::move(other.small_pool))
            , pool_size(other.pool_size) {
            other.pool_size = 0;
        }
        
        // Move assignment operator
        Bucket& operator=(Bucket&& other) noexcept {
            if (this != &other) {
                items = std::move(other.items);
                small_pool = std::move(other.small_pool);
                pool_size = other.pool_size;
                other.pool_size = 0;
            }
            return *this;
        }
        
        // Delete copy operations
        Bucket(const Bucket&) = delete;
        Bucket& operator=(const Bucket&) = delete;
    };

    struct alignas(256) Shard {
        std::vector<Bucket> buckets;
        Hash hasher;
        std::atomic<bool> is_resizing{false};
        
        explicit Shard(size_t n) : buckets(n) {}
        
        size_t getBucketIndex(size_t hashval) const {
            return hashval & (buckets.size() - 1);
        }
    };

    // Improved member variables layout
    alignas(256) std::array<std::unique_ptr<Shard>, NUM_SHARDS> shards;
    alignas(256) std::array<std::atomic<size_t>, NUM_SHARDS> shard_sizes{};

    // SIMD-optimized hash calculation for compatible key types
    template<typename T = K>
    [[nodiscard]] inline size_t calculateHash(const T& key) const {
        if constexpr (sizeof(T) == 16 && std::is_trivially_copyable_v<T>) {
            return calculateHashSIMD(key);
        } else {
            return shards[0]->hasher(key);
        }
    }

    // SIMD hash calculation for 16-byte keys
    [[nodiscard]] inline size_t calculateHashSIMD(const K& key) const {
        __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&key));
        __m128i hash = _mm_aesdec_si128(data, _mm_set1_epi32(0x2D31E867));
        return _mm_extract_epi64(hash, 0) ^ _mm_extract_epi64(hash, 1);
    }

    // Improved shard index calculation using high-quality mixing
    [[nodiscard]] inline size_t getShardIndex(size_t hashval) const {
        // MurmurHash3 finalizer for better distribution
        hashval ^= hashval >> 33;
        hashval *= 0xff51afd7ed558ccd;
        hashval ^= hashval >> 33;
        hashval *= 0xc4ceb9fe1a85ec53;
        hashval ^= hashval >> 33;
        return hashval & SHARD_MASK;
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

    // Dynamic resizing support
    void triggerResize(Shard* shard, size_t bucket_idx) {
        bool expected = false;
        if (shard->is_resizing.compare_exchange_strong(expected, true)) {
            auto& source_bucket = shard->buckets[bucket_idx];
            
            // Create two new buckets
            Bucket new_bucket1;
            Bucket new_bucket2;
            
            {
                // Lock the source bucket for exclusive access during split
                std::unique_lock lock(source_bucket.mutex);
                
                // Calculate new bucket index mask - uses 1 more bit than current
                size_t new_mask = (shard->buckets.size() << 1) - 1;
                
                // Redistribute items between new buckets based on an additional bit of their hash
                for (const auto& [key, value] : source_bucket.items) {
                    size_t item_hash = calculateHash(key);
                    if (item_hash & (shard->buckets.size())) {  // Check the next bit
                        new_bucket2.items.emplace(key, value);
                    } else {
                        new_bucket1.items.emplace(key, value);
                    }
                }
                
                // Also redistribute items from the small pool if any
                for (size_t i = 0; i < source_bucket.pool_size; ++i) {
                    const auto& [key, value] = source_bucket.small_pool[i];
                    size_t item_hash = calculateHash(key);
                    if (item_hash & (shard->buckets.size())) {
                        new_bucket2.items.emplace(key, value);
                    } else {
                        new_bucket1.items.emplace(key, value);
                    }
                }
                
                // Clear the source bucket's pool
                source_bucket.pool_size = 0;
            }
            
            // Grow the buckets vector
            size_t old_size = shard->buckets.size();
            shard->buckets.resize(old_size * 2);
            
            // Calculate new bucket positions
            size_t new_bucket_idx = bucket_idx + old_size;
            
            // Replace old bucket and insert new one
            {
                std::unique_lock lock(source_bucket.mutex);
                source_bucket.items = std::move(new_bucket1.items);
                shard->buckets[new_bucket_idx].items = std::move(new_bucket2.items);
            }
            
            shard->is_resizing.store(false);
        }
    }

public:
    // Required interface methods
    void clear() {
        for (auto& shard : shards) {
            for (auto& bucket : shard->buckets) {
                std::unique_lock lock(bucket.mutex);
                bucket.items.clear();
                bucket.pool_size = 0;
            }
        }
        for (auto& count : shard_sizes) {
            count.store(0, std::memory_order_relaxed);
        }
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

    double getLoadFactor() const {
        return static_cast<double>(size()) / capacity();
    }
};