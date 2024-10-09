/*
 * @file Allocator.h
 * @brief This file contains the definition of the MemoryPool class and the PoolAllocator class.
 *
 * This file contains the definition of the MemoryPool class, which is a custom memory pool
 * that provides memory allocation and deallocation services. The MemoryPool class is derived
 * from std::pmr::memory_resource and overrides the do_allocate and do_deallocate functions to
 * provide custom memory allocation and deallocation. The MemoryPool class also provides a
 * register_deferred_action function to register deferred actions and an invoke_deferred_actions
 * function to invoke all deferred actions.
 *
 */
#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <memory_resource> // For polymorphic memory resources
#include <memory_resource>
#include <new> // For std::bad_alloc
#include <utility>
#include <vector>

/*
 * @class MemoryPool
 * @brief A custom memory pool for memory allocation and deallocation.
 *
 * The MemoryPool class is a custom
 * memory pool that provides memory allocation and deallocation services. The MemoryPool class is derived
 * from std::pmr::memory_resource and overrides the do_allocate and do_deallocate functions to provide custom
 * memory allocation and deallocation. The MemoryPool class also provides a register_deferred_action function
 * to register deferred actions and an invoke_deferred_actions function to invoke all deferred actions.
 */
class MemoryPool : public std::pmr::memory_resource {
public:
    MemoryPool(std::size_t block_size, std::size_t block_count)
        : block_size(block_size), block_count(block_count), pool(nullptr) {
        initialize_pool();
    }

    ~MemoryPool() {
        ::operator delete(pool); // Free the entire pool
    }

    // Register a deferred action (e.g., a callback or lambda)
    void register_deferred_action(std::move_only_function<void()> action) {
        deferred_actions.push_back(std::move(action)); // Store the move-only function
    }

    // Invoke all deferred actions
    void invoke_deferred_actions() {
        for (auto &action : deferred_actions) {
            action(); // Execute the action
        }
        deferred_actions.clear(); // Clear the actions after execution
    }

protected:
    // Allocate memory from the pool
    void *do_allocate(std::size_t bytes, std::size_t alignment) override {
        if (bytes > block_size || free_list.empty()) {
            throw std::bad_alloc(); // Throw std::bad_alloc if allocation fails
        }

        // Pop the last available block from the free list
        void *result = free_list.back();
        free_list.pop_back();
        return result;
    }

    // Deallocate memory and return it to the free list
    void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override {
        if (p == nullptr) return;

        // Push the deallocated block back into the free list
        free_list.push_back(p);
    }

    bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override { return this == &other; }

private:
    std::size_t         block_size;
    std::size_t         block_count;
    void               *pool;
    std::vector<void *> free_list; // Vector-based free list for better cache locality

    // Deferred actions stored in the memory pool
    std::vector<std::move_only_function<void()>> deferred_actions;

    // Initialize the pool and populate the free list with pointers to available blocks
    void initialize_pool() {
        pool = ::operator new(block_size * block_count);

        // Populate the free list with pointers to each block
        for (std::size_t i = 0; i < block_count; ++i) {
            free_list.push_back(static_cast<char *>(pool) + i * block_size);
        }
    }
};

/*
 * @class PoolAllocator
 * @brief A custom pool allocator for memory allocation and deallocation.
 *
 * The PoolAllocator class is a custom
 * pool allocator that provides memory allocation and deallocation services. The PoolAllocator class is a
 * template class that takes a type T as a template parameter and provides the allocate and deallocate
 * functions to allocate and deallocate memory for objects of type T. The PoolAllocator class is used to
 * allocate memory from a custom memory pool, such as the MemoryPool class, and is compatible with standard
 * containers that require an allocator, such as std::vector, std::list, and std::map.
 */
template <typename T>
class PoolAllocator {
public:
    using value_type = T;

    explicit PoolAllocator(std::pmr::memory_resource *resource = std::pmr::get_default_resource())
        : resource_(resource) {}

    template <typename U>
    PoolAllocator(const PoolAllocator<U> &other) noexcept : resource_(other.resource()) {}

    T *allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) { throw std::bad_alloc(); }
        return static_cast<T *>(resource_->allocate(n * sizeof(T), alignof(T)));
    }

    void deallocate(T *p, std::size_t n) noexcept { resource_->deallocate(p, n * sizeof(T), alignof(T)); }

    template <typename U>
    bool operator==(const PoolAllocator<U> &other) const noexcept {
        return resource_ == other.resource();
    }

    template <typename U>
    bool operator!=(const PoolAllocator<U> &other) const noexcept {
        return !(*this == other);
    }

    std::pmr::memory_resource *resource() const noexcept { return resource_; }

private:
    std::pmr::memory_resource *resource_;
};
