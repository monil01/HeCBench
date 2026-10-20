#pragma once
#include "Simulator.h"
#include <cstdint>
extern "C" {
void* study_create();void study_destroy(void*);
uint64_t study_prepare(void*,uint64_t);
uint64_t study_arena_capacity(void*);uint64_t study_arena_used(void*);
Event* study_host_arena(void*);
const uint64_t* study_uploads(void*,uint64_t*);
const uint64_t* study_gate_fields(void*);
gpu_GATE* study_native_gates(void*,Event*);
uint64_t* study_output_offsets(void*);uint64_t* study_caps(void*);
void study_record_inputs(void*,const Event*,const uint64_t*);
void study_record_round(void*,const unsigned char*,const uint64_t*,uint64_t);
uint64_t study_grow(void*,const unsigned char*,uint64_t,uint64_t*);
void study_import(void*,const uint64_t*);
void study_finish_cpu(void*);
const uint64_t* study_prefix(void*,uint64_t*);
const uint64_t* study_outputs(void*,uint64_t*);
int study_validate(void*);
}
