#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void* sa_vec_new(size_t n);
void sa_vec_delete(void* handle);
int32_t* sa_vec_data(void* handle);
size_t sa_vec_size(void* handle);
int sa_vec_assign_host(void* handle,const int32_t*src,size_t n);
int sa_vec_copy_host(void* handle,int32_t*dst,size_t n);
int sa_vec_assign_vec(void*dst,void*src);
int sa_vec_get(void*handle,size_t i,int32_t*out);
int sa_sort_by_key(void*keys,void*values,size_t n);
int sa_exclusive_scan(void*values,size_t n);
int sa_runtime_version(void);
int sa_thrust_version(void);
int sa_cub_version(void);
const char*sa_last_error(void);
#ifdef __cplusplus
}
#endif
