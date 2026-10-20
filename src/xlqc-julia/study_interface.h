#pragma once
#include <cstdint>
struct XLCall {int32_t op,n;double*pbf;int32_t*map;double*D;double*out;double*Q;};
static_assert(sizeof(XLCall)==48,"XLCall ABI");
using XLCallback=int(*)(const XLCall*);
extern "C" int xl_study_main(XLCallback,int,const char*,const char*,int);
