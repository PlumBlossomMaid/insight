// insight/core/exception.h
#pragma once
#include <exception>
#include <sstream>
#include <string>

namespace ins {

    class Exception : public std::exception {
    public:
        explicit Exception(const std::string& msg, const char* file, int line) {
            std::ostringstream oss;
            oss << msg << "\n  [" << file << ":" << line << "]";
            msg_ = oss.str();
        }

        const char* what() const noexcept override {
            return msg_.c_str();
        }

    private:
        std::string msg_;
    };

    class ErrorMessage {
    public:
        template<typename... Args>
        explicit ErrorMessage(const Args&... args) {
            build(args...);
        }

        std::string to_string() const {
            return oss_.str();
        }

    private:
        void build() {}

        template<typename T>
        void build(const T& t) {
            oss_ << t;
        }

        template<typename T, typename... Args>
        void build(const T& t, const Args&... args) {
            oss_ << t;
            build(args...);
        }

        std::ostringstream oss_;
    };

} // namespace ins

#define INS_CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            auto __msg__ = ::ins::ErrorMessage(__VA_ARGS__).to_string(); \
            throw ::ins::Exception(__msg__, __FILE__, __LINE__); \
        } \
    } while(0)

#define INS_THROW(...) \
    do { \
        auto __msg__ = ::ins::ErrorMessage(__VA_ARGS__).to_string(); \
        throw ::ins::Exception(__msg__, __FILE__, __LINE__); \
    } while(0)