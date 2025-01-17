#include <atomic>
#include <memory>
#include <shared_mutex>
#include <optional>
#include <array>
#include <functional>
#include "ankerl/unordered_dense.h"

template <typename K, typename V, typename Hash = ankerl::unordered_dense::hash<K>>
class ConcurrentHashMap {
    static constexpr size_t NUM_BUCKETS = 1024;

    struct Bucket {
        ankerl::unordered_dense::map<K, V, Hash> map;
        mutable std::shared_mutex mutex;

        // Explicitly delete the copy constructor and copy assignment operator
        Bucket(const Bucket&) = delete;
        Bucket& operator=(const Bucket&) = delete;

        // Default constructor
        Bucket() = default;

        // Move constructor
        Bucket(Bucket&& other) noexcept
            : map(std::move(other.map)) {
            // Mutex is not moved; it is default-constructed
        }

        // Move assignment operator
        Bucket& operator=(Bucket&& other) noexcept {
            if (this != &other) {
                map = std::move(other.map);
                // Mutex is not moved; it is left unchanged
            }
            return *this;
        }
    };

    std::array<std::atomic<std::shared_ptr<Bucket>>, NUM_BUCKETS> buckets_;
    std::atomic<size_t> total_size_{0};

    [[nodiscard]] auto getBucketIndex(const K& key) const {
        static constexpr size_t MASK = NUM_BUCKETS - 1;
        static_assert((NUM_BUCKETS & MASK) == 0, "NUM_BUCKETS must be power of 2");
        return Hash{}(key) & MASK;
    }

public:
    auto insert(const K& key, const V& value) -> bool {
        auto& bucket_ptr = buckets_[getBucketIndex(key)];
        auto bucket = bucket_ptr.load(std::memory_order_acquire);
        {
            std::shared_lock read_lock(bucket->mutex);
            if (auto it = bucket->map.find(key); it != bucket->map.end()) {
                if (it->second == value) {
                    return false;
                }
            }
        }

        std::unique_lock write_lock(bucket->mutex);
        auto new_bucket = std::make_shared<Bucket>();
        new_bucket->map = bucket->map; // Copy the map
        auto [it, inserted] = new_bucket->map.try_emplace(key, value);
        if (!inserted) {
            it->second = value;
        } else {
            total_size_.fetch_add(1, std::memory_order_relaxed);
        }
        bucket_ptr.store(new_bucket, std::memory_order_release);
        return inserted;
    }

    [[nodiscard]] auto find(const K& key) const -> std::optional<V> {
        auto bucket = buckets_[getBucketIndex(key)].load(std::memory_order_acquire);
        std::shared_lock lock(bucket->mutex);
        if (auto it = bucket->map.find(key); it != bucket->map.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    auto erase(const K& key) -> bool {
        auto& bucket_ptr = buckets_[getBucketIndex(key)];
        auto bucket = bucket_ptr.load(std::memory_order_acquire);
        std::unique_lock write_lock(bucket->mutex);
        auto new_bucket = std::make_shared<Bucket>();
        new_bucket->map = bucket->map; // Copy the map
        if (new_bucket->map.erase(key) > 0) {
            total_size_.fetch_sub(1, std::memory_order_relaxed);
            bucket_ptr.store(new_bucket, std::memory_order_release);
            return true;
        }
        return false;
    }

    template<typename F>
    auto update(const K& key, F&& updater) -> bool {
        auto& bucket_ptr = buckets_[getBucketIndex(key)];
        auto bucket = bucket_ptr.load(std::memory_order_acquire);
        std::shared_lock read_lock(bucket->mutex);
        auto it = bucket->map.find(key);
        if (it == bucket->map.end()) {
            return false;
        }

        V new_value = it->second;
        updater(new_value);
        if (new_value == it->second) {
            return true;
        }

        read_lock.unlock();
        std::unique_lock write_lock(bucket->mutex);
        auto new_bucket = std::make_shared<Bucket>();
        new_bucket->map = bucket->map; // Copy the map
        it = new_bucket->map.find(key);
        if (it != new_bucket->map.end()) {
            it->second = std::move(new_value);
            bucket_ptr.store(new_bucket, std::memory_order_release);
            return true;
        }
        return false;
    }

    void clear() {
        for (auto& bucket_ptr : buckets_) {
            auto bucket = bucket_ptr.load(std::memory_order_acquire);
            std::unique_lock lock(bucket->mutex);
            auto new_bucket = std::make_shared<Bucket>(); // Create an empty bucket
            size_t bucket_size = bucket->map.size();
            bucket_ptr.store(new_bucket, std::memory_order_release);
            total_size_.fetch_sub(bucket_size, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] auto size() const -> size_t {
        return total_size_.load(std::memory_order_relaxed);
    }
};