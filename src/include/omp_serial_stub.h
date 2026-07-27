// Stub replacements for the OpenMP runtime functions that HeCBench benchmarks
// call from host code. When we strip #pragma omp target/parallel directives to
// produce a -serial variant, these calls remain in the source but there is no
// OpenMP runtime linked. The stubs collapse the runtime to a single virtual
// thread/team so the arithmetic that uses these ids still runs.
//
// Only defined if <omp.h> hasn't been included and OMP_SERIAL_STUB isn't
// externally overridden.
#ifndef HECBENCH_OMP_SERIAL_STUB_H
#define HECBENCH_OMP_SERIAL_STUB_H

#if !defined(_OPENMP) && !defined(_OMP_H)

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int omp_get_thread_num(void)      { return 0; }
static inline int omp_get_num_threads(void)     { return 1; }
static inline int omp_get_max_threads(void)     { return 1; }
static inline int omp_get_team_num(void)        { return 0; }
static inline int omp_get_num_teams(void)       { return 1; }
static inline int omp_in_parallel(void)         { return 0; }
static inline int omp_get_thread_limit(void)    { return 1; }
static inline int omp_get_num_procs(void)       { return 1; }
static inline int omp_get_level(void)           { return 0; }
static inline int omp_get_active_level(void)    { return 0; }
static inline int omp_is_initial_device(void)   { return 1; }
static inline int omp_get_default_device(void)  { return 0; }
static inline int omp_get_num_devices(void)     { return 0; }
static inline void omp_set_num_threads(int n)   { (void)n; }
static inline void omp_set_default_device(int d){ (void)d; }
static inline double omp_get_wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#ifdef __cplusplus
}
#endif

#endif // !_OPENMP
#endif // HECBENCH_OMP_SERIAL_STUB_H
