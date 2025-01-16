#pragma once
#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

template <typename T>
struct ParamType {
    using value_type = T;
    size_t index;
    T value;

    ParamType(T v, size_t idx = 0) : value(v), index(idx) {}
};

template <typename T>
ParamType<T> param(T value, size_t index = 0) {
    return ParamType<T>(value, index);
}

// Type traits to get the vector index for a type
template <typename T>
struct type_to_index;

template <>
struct type_to_index<double> {
    static constexpr size_t value = 0;
};

template <>
struct type_to_index<int> {
    static constexpr size_t value = 1;
};

class GenericParameters {
   private:
    std::vector<double> doubles;
    std::vector<int> ints;

   public:
    template <typename T>
    void set(const ParamType<T>& p) {
        if constexpr (std::is_same_v<T, double>) {
            if (doubles.size() <= p.index) doubles.resize(p.index + 1);
            doubles[p.index] = p.value;
        } else if constexpr (std::is_same_v<T, int>) {
            if (ints.size() <= p.index) ints.resize(p.index + 1);
            ints[p.index] = p.value;
        }
    }

    template <typename T>
    T get(size_t index) const {
        if constexpr (std::is_same_v<T, double>) {
            return index < doubles.size() ? doubles[index] : T{};
        } else if constexpr (std::is_same_v<T, int>) {
            return index < ints.size() ? ints[index] : T{};
        }
        return T{};
    }
};

class ParametersBuilder {
   private:
    GenericParameters params;

   public:
    ParametersBuilder(ParamType<double> initial_cost, ParamType<int> distance,
                      ParamType<double> node_id = param(0.0),
                      ParamType<double> resources = param(0.0)) {
        params.set(initial_cost);
        params.set(distance);
    }

    template <typename T>
    ParametersBuilder& with(const ParamType<T>& p) {
        params.set(p);
        return *this;
    }

    GenericParameters build() { return params; }
};

class CostFunction {
   private:
    using CostFun = std::function<double(const GenericParameters&)>;
    CostFun cost_func;

   public:
    CostFunction()
        : cost_func([](const GenericParameters& params) {
              return params.get<double>(0) + params.get<int>(0);
          }) {}

    explicit CostFunction(const CostFun& custom_func)
        : cost_func(custom_func) {}

    void set_cost_function(const CostFun& new_cost_func) {
        if (!new_cost_func) {
            throw std::invalid_argument("Cost function cannot be null");
        }
        cost_func = new_cost_func;
    }

    double calculate_cost(const GenericParameters& params) const {
        if (!cost_func) {
            throw std::runtime_error("No cost function set");
        }
        return cost_func(params);
    }

    double calculate_cost(double initial_cost, int distance,
                          double node_id = 0.0,
                          std::vector<double> resources = {}) {
        auto params =
            ParametersBuilder(param<double>(initial_cost), param<int>(distance))
                .build();
        return calculate_cost(params);
    }
};
