// MSVC compatibility: aligned allocation helpers
// Replaces placement new with std::align_val_t which MSVC rejects (C2956)

#pragma once

#include <cstdlib>
#include <new>

#ifdef _MSC_VER

template<typename T>
T* aligned_new_array(std::align_val_t alignment, size_t count) {
    void* ptr = _aligned_malloc(count * sizeof(T), static_cast<size_t>(alignment));
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
}

template<typename T>
void aligned_delete_array(T* ptr) {
    _aligned_free(ptr);
}

#else

template<typename T>
T* aligned_new_array(std::align_val_t alignment, size_t count) {
    return new(alignment) T[count];
}

template<typename T>
void aligned_delete_array(T* ptr) {
    delete[] ptr;
}

#endif
