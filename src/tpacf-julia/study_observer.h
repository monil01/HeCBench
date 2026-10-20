#pragma once
#include <cstddef>
#include <cstdint>
extern "C" {
void tp_start(const char*,const char*,const char*);
void tp_begin(int);
void tp_case(int,int,int,int,const int*,int,const double*);
void tp_input(const unsigned char*,size_t);
void tp_guard(int,const unsigned char*);
void tp_output(const int64_t*,const int64_t*,const char*);
void tp_write(int,double);
void tp_finish();
}
