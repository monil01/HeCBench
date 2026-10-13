#pragma once
#include <iris/iris.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <cstdint>

namespace port {
inline void check(int status) {
    if (status != IRIS_SUCCESS) { fprintf(stderr, "IRIS error %d\n", status); exit(2); }
}
struct Buffer {
    void* host;
    size_t bytes;
    std::unique_ptr<iris::DMem> mem;
};
inline std::map<uintptr_t, Buffer> buffers;
inline int device = -1;
inline auto locate(const void* ptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    auto it = buffers.upper_bound(addr);
    if (it == buffers.begin()) { fprintf(stderr, "Unknown DMem pointer\n"); exit(2); }
    --it;
    if (addr >= it->first + it->second.bytes) { fprintf(stderr, "DMem offset out of range\n"); exit(2); }
    return it;
}
struct Session {
    iris::Platform platform;
    Session(int* argc, char*** argv) {
        check(platform.init(argc, argv, 1));
        int count = 0; check(iris_device_count(&count));
        for (int i = 0; i < count; ++i) {
            int backend = 0, type = 0; char name[256] = {};
            check(iris_device_info(i, iris_backend, &backend, nullptr));
            check(iris_device_info(i, iris_type, &type, nullptr));
            check(iris_device_info(i, iris_name, name, nullptr));
            printf("IRIS device %d name=%s backend=%d type=%d\n", i, name, backend, type);
            if (device < 0 && backend == iris_hip && (type & iris_amd)) device = i;
        }
        if (device < 0) { fprintf(stderr, "blocked: no-amd-gpu HIP device\n"); exit(2); }
        printf("Selected IRIS HIP AMD device %d\n", device);
    }
    ~Session() {
        for (auto& kv : buffers) { kv.second.mem.reset(); free(kv.second.host); }
        buffers.clear();
        check(platform.finalize());
    }
};
template<typename P> inline int allocate(P* ptr, size_t bytes) {
    void* host = calloc(1, bytes);
    if (!host) { fprintf(stderr, "Host allocation failed\n"); exit(2); }
    *ptr = reinterpret_cast<std::remove_reference_t<decltype(*ptr)>>(host);
    buffers.emplace(reinterpret_cast<uintptr_t>(host), Buffer{host, bytes, std::make_unique<iris::DMem>(host, bytes)});
    return IRIS_SUCCESS;
}
inline int allocate(void** ptr, size_t bytes) {
    void* host = calloc(1, bytes);
    if (!host) exit(2);
    *ptr = host;
    buffers.emplace(reinterpret_cast<uintptr_t>(host), Buffer{host, bytes, std::make_unique<iris::DMem>(host, bytes)});
    return IRIS_SUCCESS;
}
inline int flush(Buffer& b) {
    iris::Task task; check(task.flush_out(*b.mem)); check(task.submit(device, nullptr, true));
    return IRIS_SUCCESS;
}
enum Direction { h2d, d2h, d2d };
inline int copy(void* dst, const void* src, size_t bytes, Direction direction) {
    if (direction == d2h || direction == d2d) { auto it = locate(src); check(flush(it->second)); }
    memmove(dst, src, bytes);
    if (direction == h2d || direction == d2d) {
        auto it = locate(dst); check(it->second.mem->update(it->second.host));
    }
    return IRIS_SUCCESS;
}
inline int zero(void* ptr, int value, size_t bytes) {
    auto it = locate(ptr);
    memset(ptr, value, bytes);
    check(it->second.mem->update(it->second.host));
    return IRIS_SUCCESS;
}
inline int release(void* ptr) {
    auto it = buffers.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == buffers.end()) exit(2);
    it->second.mem.reset(); free(it->second.host); buffers.erase(it);
    return IRIS_SUCCESS;
}
template<typename P> inline int host_allocate(P* ptr, size_t bytes) {
    *ptr = reinterpret_cast<std::remove_reference_t<decltype(*ptr)>>(malloc(bytes));
    return *ptr ? IRIS_SUCCESS : IRIS_ERROR;
}
inline int host_allocate(void** ptr, size_t bytes) { *ptr = malloc(bytes); return *ptr ? IRIS_SUCCESS : IRIS_ERROR; }
inline int host_release(void* ptr) { free(ptr); return IRIS_SUCCESS; }
template<typename T> inline void argument(iris_kernel kernel, int index, T& value) {
    if constexpr (std::is_pointer_v<T>) {
        if (!value) { check(iris_kernel_setarg(kernel,index,sizeof(T),&value)); return; }
        auto ptr = const_cast<const void*>(static_cast<const volatile void*>(value));
        auto it = locate(ptr);
        size_t offset = reinterpret_cast<uintptr_t>(ptr) - it->first;
        int mode = std::is_const_v<std::remove_pointer_t<T>> ? iris_r : iris_rw;
        check(iris_kernel_setmem_off(kernel, index, it->second.mem->mem(), offset, mode));
    } else {
        check(iris_kernel_setarg(kernel, index, sizeof(T), &value));
    }
}
template<typename... A> inline void launch(const char* name, dim3 grid, dim3 block, A... args) {
    iris_kernel kernel; iris_task task;
    check(iris_kernel_create(name, &kernel));
    auto values = std::make_tuple(args...);
    int index = 0;
    std::apply([&](auto&... x) { (argument(kernel, index++, x), ...); }, values);
    size_t gws[] = {size_t(grid.x)*block.x, size_t(grid.y)*block.y, size_t(grid.z)*block.z};
    size_t lws[] = {block.x, block.y, block.z};
    check(iris_task_create(&task));
    check(iris_task_kernel_object(task, kernel, 3, nullptr, gws, lws));
    check(iris_task_submit(task, device, nullptr, 1));
    check(iris_kernel_release(kernel));
}
}
