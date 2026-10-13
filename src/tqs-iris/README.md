# Task queue synchronization with IRIS DMem

Preserves original parameter defaults, enqueue logic, work/inactive task math,
atomic queue scheduling and reference checks. Shared queue scratch is a static
three-int array matching original sizeof(int)+sizeof(task_t). Bounds guards
avoid original reads of queue[next] when next>=gpuQueueSize; task results and
coverage unchanged. DMem.update republishes each host queue/reset counter.
Full pool checking covers inactive tasks skipped by the source verifier.
Optional HECBENCH_TQS_OUTPUT dumps the entire final data pool. Native isolated
adapter adds the identical full verifier/dump only and missing support headers.
Real original CUDA pattern input is used; RUN_ARGS points at that sibling input.
All source argument/default values otherwise preserved.
