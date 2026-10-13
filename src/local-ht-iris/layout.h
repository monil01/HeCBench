#pragma once
#include <cstdint>
#define LASSM_MAX_KMER_LEN 121
struct cstr_type{char* start_ptr;int length;};
struct ExtCounts{uint32_t count_A,count_C,count_G,count_T;};
struct MerFreqs{ExtCounts hi_q_exts,low_q_exts;char ext;int count;};
struct loc_ht{cstr_type key;MerFreqs val;};
struct loc_ht_bool{cstr_type key;bool val;};
static_assert(sizeof(cstr_type)==16 && sizeof(loc_ht)==56 && sizeof(loc_ht_bool)==24,"hash table ABI");
