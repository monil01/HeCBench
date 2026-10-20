#pragma once
#include <cstddef>
#include <cstdint>
extern "C" {void snap_start(const char*,const char*,const char*);void snap_begin(int);void snap_case(int,const uint32_t*,const double*,int);void snap_input(int,int,int,const unsigned char*,size_t);void snap_guard(int,int,int,const unsigned char*);void snap_output(int,int,const double*,const double*);void snap_case_end(int,double);void snap_write(int,double);void snap_finish();}
