#pragma once

#include <chrono>

namespace g13 {

template <typename Func, typename... Args, class TimeDuration = std::chrono::nanoseconds>
static auto measureTime(Func func, Args &&...args) {
    using FuncReturnType = decltype(func(std::forward<Args>(args)...));

    const auto start = std::chrono::high_resolution_clock::now();
    if constexpr (std::is_same_v<void, FuncReturnType>) {
        func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<TimeDuration>(end - start).count();
    } else {
        FuncReturnType ret = func(std::forward<Args>(args)...);
        auto           end = std::chrono::high_resolution_clock::now();
        return std::tuple<FuncReturnType, uint64_t>{
            ret, std::chrono::duration_cast<TimeDuration>(end - start).count()};
    }
}

}