#pragma once
#include <iris/iris.hpp>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
#include "backend.h"
inline std::unordered_map<iris::DMem*,void*> storage;
inline iris::DMem *create_mem(size_t n) { void *h=calloc(1,n); auto *m=new iris::DMem(h,n); storage[m]=h; return m; }
inline void copy_to(iris::DMem *m,const void *h,size_t n) { memcpy(storage.at(m),h,n); m->update(storage.at(m)); }
inline void clear_mem(iris::DMem *m,size_t n) { memset(storage.at(m),0,n); m->update(storage.at(m)); }
inline void release_mem(iris::DMem *m) { auto h=storage.at(m); delete m; free(h); storage.erase(m); }
