#include "gpu_lock.hpp"
#include <stdio.h>
#include <iris/iris.hpp>
#include "./iris_wrappers.h"

extern iris::Platform *g_iris_platform;

static size_t work_size(int count, int order) {
  int threads = order < 256 ? order : 256;
  return static_cast<size_t>(count) * static_cast<size_t>(threads);
}

static size_t local_size(int order) {
  return static_cast<size_t>(order < 256 ? order : 256);
}

void kernel_wrapper(record *records,
                    long records_mem,
                    knode *knodes,
                    long knodes_elem,
                    long knodes_mem,
                    int order,
                    long maxheight,
                    int count,
                    long *currKnode,
                    long *offset,
                    int *keys,
                    record *ans) {
  GpuLock lock;
  printf("# of blocks = %d, # of threads/block = %zu (IRIS dmem HIP)\n",
         count, local_size(order));
  iris::DMem mem_records(records, static_cast<size_t>(records_mem));
  iris::DMem mem_knodes(knodes, static_cast<size_t>(knodes_mem));
  iris::DMem mem_curr(currKnode, static_cast<size_t>(count) * sizeof(long));
  iris::DMem mem_offset(offset, static_cast<size_t>(count) * sizeof(long));
  iris::DMem mem_keys(keys, static_cast<size_t>(count) * sizeof(int));
  iris::DMem mem_ans(ans, static_cast<size_t>(count) * sizeof(record));

  iris::Task task;
  void *params[] = { &maxheight, &mem_knodes, &knodes_elem, &mem_records,
                     &mem_curr, &mem_offset, &mem_keys, &mem_ans };
  int pinfo[] = { sizeof(maxheight), iris_r, sizeof(knodes_elem), iris_r,
                  iris_rw, iris_rw, iris_r, iris_rw };
  size_t gws[] = { work_size(count, order) };
  size_t lws[] = { local_size(order) };
  task.kernel("findK", 1, NULL, gws, lws, 8, params, pinfo);
  task.flush_out(mem_curr);
  task.flush_out(mem_offset);
  task.flush_out(mem_ans);
  task.submit(iris_gpu, NULL, 1);
}

void kernel2_wrapper(knode *knodes,
                     long knodes_elem,
                     long knodes_mem,
                     int order,
                     long maxheight,
                     int count,
                     long *currKnode,
                     long *offset,
                     long *lastKnode,
                     long *offset_2,
                     int *start,
                     int *end,
                     int *recstart,
                     int *reclength) {
  GpuLock lock;
  printf("# of blocks = %d, # of threads/block = %zu (IRIS dmem HIP)\n",
         count, local_size(order));
  iris::DMem mem_knodes(knodes, static_cast<size_t>(knodes_mem));
  iris::DMem mem_curr(currKnode, static_cast<size_t>(count) * sizeof(long));
  iris::DMem mem_offset(offset, static_cast<size_t>(count) * sizeof(long));
  iris::DMem mem_last(lastKnode, static_cast<size_t>(count) * sizeof(long));
  iris::DMem mem_offset2(offset_2, static_cast<size_t>(count) * sizeof(long));
  iris::DMem mem_start(start, static_cast<size_t>(count) * sizeof(int));
  iris::DMem mem_end(end, static_cast<size_t>(count) * sizeof(int));
  iris::DMem mem_recstart(recstart, static_cast<size_t>(count) * sizeof(int));
  iris::DMem mem_reclength(reclength, static_cast<size_t>(count) * sizeof(int));

  iris::Task task;
  void *params[] = { &maxheight, &mem_knodes, &knodes_elem, &mem_curr,
                     &mem_offset, &mem_last, &mem_offset2, &mem_start,
                     &mem_end, &mem_recstart, &mem_reclength };
  int pinfo[] = { sizeof(maxheight), iris_r, sizeof(knodes_elem), iris_rw,
                  iris_rw, iris_rw, iris_rw, iris_r, iris_r, iris_rw, iris_rw };
  size_t gws[] = { work_size(count, order) };
  size_t lws[] = { local_size(order) };
  task.kernel("findRangeK", 1, NULL, gws, lws, 11, params, pinfo);
  task.flush_out(mem_curr);
  task.flush_out(mem_offset);
  task.flush_out(mem_last);
  task.flush_out(mem_offset2);
  task.flush_out(mem_recstart);
  task.flush_out(mem_reclength);
  task.submit(iris_gpu, NULL, 1);
}
