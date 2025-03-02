#pragma once

#include <concepts>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace serializer {

// ---------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------
class SerializationError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------
// Type traits and concepts
// ---------------------------------------------------------------------
template <typename T>
concept Reflectable = requires(const T& t) {
    {
        t.visit([](const char*, const auto&) {})
    } -> std::same_as<void>;
};

template <typename T>
concept String_Like =
    std::same_as<T, std::string> || std::same_as<T, std::string_view>;

template <typename T>
concept Array_Like = requires(T t) {
    typename T::value_type;
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.begin() } -> std::input_iterator;
    { t.end() } -> std::input_iterator;
    { std::tuple_size<T>::value } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept Container = requires(T t) {
    typename T::value_type;
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.begin() } -> std::input_iterator;
    { t.end() } -> std::input_iterator;
} && !Array_Like<T> && !String_Like<T>;

template <typename T>
concept Trivially_Serializable =
    std::is_trivially_copyable_v<T> && !Array_Like<T> && !Container<T> &&
    !String_Like<T> && !Reflectable<T>;

// ---------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------
template <typename T>
void serialize(const T& obj, std::ostream& os);

template <typename T>
void deserialize(T& obj, std::istream& is);

template <Array_Like T>
void serialize_value(const T& arr, std::ostream& os);

template <Array_Like T>
void deserialize_value(T& arr, std::istream& is);

template <Trivially_Serializable T>
void serialize_value(const T& value, std::ostream& os);

template <Trivially_Serializable T>
void deserialize_value(T& value, std::istream& is);

template <String_Like T>
void serialize_value(const T& str, std::ostream& os);

template <String_Like T>
void deserialize_value(T& str, std::istream& is);

template <Container T>
void serialize_value(const T& container, std::ostream& os);

template <Container T>
void deserialize_value(T& container, std::istream& is);

template <typename T>
    requires Reflectable<T>
void serialize_value(const T& value, std::ostream& os);

template <typename T>
    requires Reflectable<T>
void deserialize_value(T& value, std::istream& is);

// ---------------------------------------------------------------------
// Serialization buffer
// ---------------------------------------------------------------------
class SerializationBuffer {
    static constexpr std::size_t BUFFER_SIZE = 8192;
    std::vector<char> buffer_;
    std::size_t pos_ = 0;

   public:
    SerializationBuffer() : buffer_(BUFFER_SIZE) {}

    void flush(std::ostream& os) {
        if (pos_ > 0) {
            os.write(buffer_.data(), pos_);
            pos_ = 0;
        }
    }

    void write(const char* data, size_t size, std::ostream& os) {
        if (pos_ + size > BUFFER_SIZE) {
            flush(os);
            if (size > BUFFER_SIZE) {
                os.write(data, size);
                return;
            }
        }
        std::memcpy(buffer_.data() + pos_, data, size);
        pos_ += size;
    }
};

// ---------------------------------------------------------------------
// Reflection macros
// ---------------------------------------------------------------------
#define REFLECT_MEMBER(member) f(#member, member);

#define REFLECT(...)                          \
    template <typename F>                     \
    void visit(F&& f) {                       \
        FOR_EACH(REFLECT_MEMBER, __VA_ARGS__) \
    }                                         \
    template <typename F>                     \
    void visit(F&& f) const {                 \
        FOR_EACH(REFLECT_MEMBER, __VA_ARGS__) \
    }

#define FOR_EACH_1(what, x) what(x)
#define FOR_EACH_2(what, x, ...) what(x) FOR_EACH_1(what, __VA_ARGS__)
#define FOR_EACH_3(what, x, ...) what(x) FOR_EACH_2(what, __VA_ARGS__)
#define FOR_EACH_4(what, x, ...) what(x) FOR_EACH_3(what, __VA_ARGS__)
#define FOR_EACH_5(what, x, ...) what(x) FOR_EACH_4(what, __VA_ARGS__)
#define FOR_EACH_6(what, x, ...) what(x) FOR_EACH_5(what, __VA_ARGS__)
#define FOR_EACH_7(what, x, ...) what(x) FOR_EACH_6(what, __VA_ARGS__)

#define GET_FOR_EACH_MACRO(_1, _2, _3, _4, _5, _6, _7, NAME, ...) NAME
#define FOR_EACH(macro, ...)                                            \
    GET_FOR_EACH_MACRO(__VA_ARGS__, FOR_EACH_7, FOR_EACH_6, FOR_EACH_5, \
                       FOR_EACH_4, FOR_EACH_3, FOR_EACH_2, FOR_EACH_1)  \
    (macro, __VA_ARGS__)

// ---------------------------------------------------------------------
// Core serialization implementations
// ---------------------------------------------------------------------
template <Array_Like T>
void serialize_value(const T& arr, std::ostream& os) {
    for (const auto& item : arr) {
        serialize_value(item, os);
    }
}

template <Array_Like T>
void deserialize_value(T& arr, std::istream& is) {
    for (auto& item : arr) {
        deserialize_value(item, is);
    }
}

template <Trivially_Serializable T>
void serialize_value(const T& value, std::ostream& os) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <Trivially_Serializable T>
void deserialize_value(T& value, std::istream& is) {
    if (!is.read(reinterpret_cast<char*>(&value), sizeof(T))) {
        throw SerializationError("Failed to read trivial type");
    }
}

template <String_Like T>
void serialize_value(const T& str, std::ostream& os) {
    const size_t size = str.size();
    serialize_value(size, os);
    os.write(str.data(), size);
}

template <String_Like T>
void deserialize_value(T& str, std::istream& is) {
    size_t size;
    deserialize_value(size, is);
    str.resize(size);
    if (!is.read(&str[0], size)) {
        throw SerializationError("Failed to read string data");
    }
}

template <Container T>
void serialize_value(const T& container, std::ostream& os) {
    const size_t size = container.size();
    serialize_value(size, os);
    for (const auto& item : container) {
        serialize_value(item, os);
    }
}

template <Container T>
void deserialize_value(T& container, std::istream& is) {
    size_t size;
    deserialize_value(size, is);
    container.clear();
    if constexpr (requires { container.reserve(size); }) {
        container.reserve(size);
    }
    for (size_t i = 0; i < size; ++i) {
        typename T::value_type item;
        deserialize_value(item, is);
        container.insert(container.end(), std::move(item));
    }
}

template <typename T>
    requires Reflectable<T>
void serialize_value(const T& value, std::ostream& os) {
    serialize(value, os);
}

template <typename T>
    requires Reflectable<T>
void deserialize_value(T& value, std::istream& is) {
    deserialize(value, is);
}

// ---------------------------------------------------------------------
// Generic serialization using reflection
// ---------------------------------------------------------------------
template <typename T>
void serialize(const T& obj, std::ostream& os) {
    obj.visit([&os](const char* name, const auto& member) {
        size_t nameLen = std::strlen(name);
        serialize_value(nameLen, os);
        os.write(name, nameLen);
        serialize_value(member, os);
    });
}

template <typename T>
void deserialize(T& obj, std::istream& is) {
    obj.visit([&is](const char* /*name*/, auto& member) {
        size_t nameLen;
        deserialize_value(nameLen, is);
        std::string dummy(nameLen, '\0');
        is.read(&dummy[0], nameLen);
        deserialize_value(member, is);
    });
}

// ---------------------------------------------------------------------
// File I/O wrapper
// ---------------------------------------------------------------------
template <typename T>
class Serializer {
   public:
    static void toFile(const T& obj, const std::string& filename) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            throw SerializationError("Could not open file for writing: " +
                                     filename);
        }
        serialize(obj, ofs);
    }

    static void fromFile(T& obj, const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) {
            throw SerializationError("Could not open file for reading: " +
                                     filename);
        }
        deserialize(obj, ifs);
    }
};

}  // namespace serializer
