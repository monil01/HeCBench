#pragma once
#include <cstdint>
extern "C" {
void* pad_create(int m,int n,int padding,float alpha,const uint64_t*initial);
void pad_destroy(void*);
double* pad_data(void*);
double* pad_all_data(void*);
int* pad_flags(void*);
void pad_reset(void*);
void pad_start(void*);
void pad_join(void*);
}
