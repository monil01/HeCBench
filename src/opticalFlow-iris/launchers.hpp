#pragma once
#include "dmem.hpp"
#include "common.h"
static void Downscale(FlowBuffer &src, int width, int height, int stride,
                      int newWidth, int newHeight, int newStride, FlowBuffer &out) {
    void *args[] = { &newWidth, &newHeight, &newStride, &out.mem, &src.mem, &width, &height, &stride };
  int info[] = { sizeof(newWidth), sizeof(newHeight), sizeof(newStride), iris_rw, iris_r, sizeof(width), sizeof(height), sizeof(stride) };
  size_t gws[] = { size_t(iDivUp(newWidth,32) * 32), size_t(iDivUp(newHeight,8) * 8) };
  size_t lws[] = { 32,8 };
  iris::Task task;
  iris_check(task.kernel("DownscaleKernel", 2, nullptr, gws, lws, 8, args, info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}

static void Upscale(FlowBuffer &src, int width, int height, int stride,
                    int newWidth, int newHeight, int newStride,
                    float scale, FlowBuffer &out) {
    void *args[] = { &newWidth, &newHeight, &newStride, &scale, &out.mem, &src.mem, &width, &height, &stride };
  int info[] = { sizeof(newWidth), sizeof(newHeight), sizeof(newStride), sizeof(scale), iris_rw, iris_r, sizeof(width), sizeof(height), sizeof(stride) };
  size_t gws[] = { size_t(iDivUp(newWidth,32) * 32), size_t(iDivUp(newHeight,8) * 8) };
  size_t lws[] = { 32,8 };
  iris::Task task;
  iris_check(task.kernel("UpscaleKernel", 2, nullptr, gws, lws, 9, args, info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}

static void WarpImage(FlowBuffer &src, int w, int h, int s,
                      FlowBuffer &u, FlowBuffer &v, FlowBuffer &out) {
    void *args[] = { &w, &h, &s, &u.mem, &v.mem, &out.mem, &src.mem, &w, &h, &s };
  int info[] = { sizeof(w), sizeof(h), sizeof(s), iris_r, iris_r, iris_rw, iris_r, sizeof(w), sizeof(h), sizeof(s) };
  size_t gws[] = { size_t(iDivUp(w,32) * 32), size_t(iDivUp(h,6) * 6) };
  size_t lws[] = { 32,6 };
  iris::Task task;
  iris_check(task.kernel("WarpingKernel", 2, nullptr, gws, lws, 10, args, info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}

static void ComputeDerivatives(FlowBuffer &I0, FlowBuffer &I1,
                               int w, int h, int s,
                               FlowBuffer &Ix, FlowBuffer &Iy, FlowBuffer &Iz) {
    void *args[] = { &w, &h, &s, &Ix.mem, &Iy.mem, &Iz.mem, &I0.mem, &w, &h, &s, &I1.mem, &w, &h, &s };
  int info[] = { sizeof(w), sizeof(h), sizeof(s), iris_rw, iris_rw, iris_rw, iris_r, sizeof(w), sizeof(h), sizeof(s), iris_r, sizeof(w), sizeof(h), sizeof(s) };
  size_t gws[] = { size_t(iDivUp(w,32) * 32), size_t(iDivUp(h,6) * 6) };
  size_t lws[] = { 32,6 };
  iris::Task task;
  iris_check(task.kernel("ComputeDerivativesKernel", 2, nullptr, gws, lws, 14, args, info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}

static void SolveForUpdate(FlowBuffer &du0, FlowBuffer &dv0, FlowBuffer &Ix,
                           FlowBuffer &Iy, FlowBuffer &Iz, int w, int h,
                           int s, float alpha, FlowBuffer &du1, FlowBuffer &dv1) {
  // CTA size
  // grid size

    void *args[] = { &du0.mem, &dv0.mem, &Ix.mem, &Iy.mem, &Iz.mem, &w, &h, &s, &alpha, &du1.mem, &dv1.mem };
  int info[] = { iris_r, iris_r, iris_rw, iris_rw, iris_rw, sizeof(w), sizeof(h), sizeof(s), sizeof(alpha), iris_rw, iris_rw };
  size_t gws[] = { size_t(iDivUp(w,32) * 32), size_t(iDivUp(h,6) * 6) };
  size_t lws[] = { 32,6 };
  iris::Task task;
  iris_check(task.kernel("JacobiIteration", 2, nullptr, gws, lws, 11, args, info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}

static void Add(FlowBuffer &op1, FlowBuffer &op2, int count, FlowBuffer &sum) {

    void *args[] = { &op1.mem, &op2.mem, &count, &sum.mem };
  int info[] = { iris_r, iris_r, sizeof(count), iris_rw };
  size_t gws[] = { size_t(iDivUp(count,256) * 256), size_t(iDivUp(1,1) * 1) };
  size_t lws[] = { 256,1 };
  iris::Task task;
  iris_check(task.kernel("AddKernel", 1, nullptr, gws, lws, 4, args, info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}
