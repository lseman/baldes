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
#include <span>
#include <bit>

template <typename K, typename V, typename Hash = std::hash<K>>
class ConcurrentHashMap {
public:
    static constexpr size_t NUM_SHARDS = 128;
    
    explicit ConcurrentHashMap(size_t bucket_count = 128) 
        : total_size_(0) {
        size_t per_shard = std::max(size_t(128), bucket_count / NUM_SHARDS);
        per_shard = std::bit_ceil(per_shard);
        for (auto& shard : shards_) {
            shard = std::make_unique<Shard>(per_shard);
        }
    }

    bool insert(const K& key, V value) {
        return insert_impl(key, std::move(value));
    }

    bool insert(K&& key, V value) {
        return insert_impl(std::move(key), std::move(value));
    }

    template<typename... Args>
    bool emplace(const K& key, Args&&... args) {
        return insert_impl(key, V(std::forward<Args>(args)...));
    }

    std::optional<V> find(const K& key) const {
        const size_t hashval = calculateHash(key);
        const auto& shard = shards_[getShardIndex(hashval)];
        const size_t bucket_idx = shard->getBucketIndex(hashval);
        const auto& bucket = shard->buckets[bucket_idx];

        for (int retry = 0; retry < 3; ++retry) {
            std::shared_lock lock(bucket.mutex, std::try_to_lock);
            if (lock.owns_lock()) {
                auto it = bucket.items.find(key);
                if (it != bucket.items.end()) {
                    return it->second;
                }
                return std::nullopt;
            }
            for (int i = 0; i < (1 << retry); ++i) {
                _mm_pause();
            }
        }
        return std::nullopt;
    }

    void clear() {
        for (auto& shard : shards_) {
            for (auto& bucket : shard->buckets) {
                std::unique_lock lock(bucket.mutex);
                bucket.items.clear();
                bucket.pool_size = 0;
            }
        }
        total_size_.store(0, std::memory_order_relaxed);
    }

    template<typename F>
    bool update(const K& key, F&& updater) {
        const size_t hashval = calculateHash(key);
        auto& shard = shards_[getShardIndex(hashval)];
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
        return total_size_.load(std::memory_order_relaxed);
    }

    size_t capacity() const {
        size_t total = 0;
        for (const auto& shard : shards_) {
            total += shard->buckets.size();
        }
        return total;
    }

private:
    static constexpr size_t SHARD_MASK = NUM_SHARDS - 1;
    static constexpr size_t BUCKET_RESIZE_THRESHOLD = 32;

    struct alignas(64) Bucket {
        std::unordered_map<K, V, Hash> items;
        mutable std::shared_mutex mutex;
        
        static constexpr size_t POOL_SIZE = 16;
        struct {
            alignas(K) std::byte key_storage[sizeof(K)];
            alignas(V) std::byte value_storage[sizeof(V)];
            bool occupied{false};
        } pool[POOL_SIZE];
        std::atomic<size_t> pool_size{0};

        Bucket() = default;
        
        Bucket(Bucket&& other) noexcept 
            : items(std::move(other.items))
            , pool_size(other.pool_size.load(std::memory_order_relaxed)) {
            memcpy(pool, other.pool, sizeof(pool));
        }
        
        Bucket& operator=(Bucket&& other) noexcept {
            if (this != &other) {
                items = std::move(other.items);
                pool_size.store(other.pool_size.load(std::memory_order_relaxed));
                memcpy(pool, other.pool, sizeof(pool));
            }
            return *this;
        }

        ~Bucket() {
            for (size_t i = 0; i < pool_size.load(std::memory_order_relaxed); ++i) {
                if (pool[i].occupied) {
                    std::destroy_at(reinterpret_cast<K*>(&pool[i].key_storage));
                    std::destroy_at(reinterpret_cast<V*>(&pool[i].value_storage));
                }
            }
        }

        Bucket(const Bucket&) = delete;
        Bucket& operator=(const Bucket&) = delete;
    };

    struct alignas(64) Shard {
        std::vector<Bucket> buckets;
        Hash hasher;
        std::atomic<bool> is_resizing{false};
        
        explicit Shard(size_t n) : buckets(n) {}
        
        size_t getBucketIndex(size_t hashval) const {
            return hashval & (buckets.size() - 1);
        }
    };

    alignas(64) std::array<std::unique_ptr<Shard>, NUM_SHARDS> shards_;
    alignas(64) std::atomic<size_t> total_size_;

    [[nodiscard]] inline size_t getShardIndex(size_t hashval) const {
        hashval ^= hashval >> 33;
        hashval *= 0xff51afd7ed558ccd;
        hashval ^= hashval >> 33;
        hashval *= 0xc4ceb9fe1a85ec53;
        hashval ^= hashval >> 33;
        return hashval & SHARD_MASK;
    }

    template<typename T = K>
    [[nodiscard]] inline size_t calculateHash(const T& key) const {
        if constexpr (sizeof(T) == 16 && std::is_trivially_copyable_v<T> && 
                     std::is_standard_layout_v<T>) {
            return calculateHashSIMD(key);
        } else {
            return shards_[0]->hasher(key);
        }
    }

    [[nodiscard]] inline size_t calculateHashSIMD(const K& key) const {
        __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&key));
        __m128i hash = _mm_aesdec_si128(data, _mm_set1_epi32(0x2D31E867));
        hash = _mm_aesdec_si128(hash, _mm_set1_epi32(0x7E624B47));
        return _mm_extract_epi64(hash, 0) ^ _mm_extract_epi64(hash, 1);
    }

    template<typename KeyArg>
    bool insert_impl(KeyArg&& key, V value) {
        const size_t hashval = calculateHash(key);
        auto& shard = shards_[getShardIndex(hashval)];
        const size_t bucket_idx = shard->getBucketIndex(hashval);
        auto& bucket = shard->buckets[bucket_idx];

        {
            std::shared_lock lock(bucket.mutex, std::try_to_lock);
            if (lock.owns_lock()) {
                if (bucket.items.find(key) != bucket.items.end()) {
                    return false;
                }
            }
        }

        std::unique_lock lock(bucket.mutex);
        auto [it, inserted] = bucket.items.try_emplace(
            std::forward<KeyArg>(key), std::move(value));
        
        if (inserted) {
            total_size_.fetch_add(1, std::memory_order_relaxed);
            if (bucket.items.size() > BUCKET_RESIZE_THRESHOLD) {
                triggerResize(shard.get(), bucket_idx);
            }
        }
        return inserted;
    }

    void triggerResize(Shard* shard, size_t bucket_idx) {
        bool expected = false;
        if (shard->is_resizing.compare_exchange_strong(expected, true, 
                                                     std::memory_order_acquire,
                                                     std::memory_order_relaxed)) {
            try {
                auto& source_bucket = shard->buckets[bucket_idx];
                
                Bucket new_bucket1;
                Bucket new_bucket2;
                
                {
                    std::unique_lock lock(source_bucket.mutex);
                    
                    for (const auto& [key, value] : source_bucket.items) {
                        size_t item_hash = calculateHash(key);
                        if (item_hash & (shard->buckets.size())) {
                            new_bucket2.items.emplace(key, value);
                        } else {
                            new_bucket1.items.emplace(key, value);
                        }
                    }
                    
                    source_bucket.pool_size.store(0, std::memory_order_relaxed);
                }
                
                size_t old_size = shard->buckets.size();
                shard->buckets.resize(old_size * 2);
                
                size_t new_bucket_idx = bucket_idx + old_size;
                
                {
                    std::unique_lock lock(source_bucket.mutex);
                    source_bucket.items = std::move(new_bucket1.items);
                    shard->buckets[new_bucket_idx].items = std::move(new_bucket2.items);
                }
            }
            catch (...) {
                shard->is_resizing.store(false, std::memory_order_release);
                throw;
            }
            
            shard->is_resizing.store(false, std::memory_order_release);
        }
    }
};