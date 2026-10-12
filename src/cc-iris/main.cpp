#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>

#include "../mis-cuda/graph.h"

static const int ThreadsPerBlock = 256;

static void launch_kernel(const char* name,
                          int nodes,
                          iris::DMem& mem_nidx,
                          iris::DMem& mem_nlist,
                          iris::DMem& mem_nstat,
                          iris::DMem* mem_wl,
                          int blocks)
{
  size_t gws[1] = {static_cast<size_t>(blocks) * ThreadsPerBlock};
  size_t lws[1] = {ThreadsPerBlock};
  iris::Task task;
  if (mem_wl == nullptr) {
    void* params[4] = {&nodes, &mem_nidx, &mem_nlist, &mem_nstat};
    int pinfo[4] = {sizeof(nodes), iris_r, iris_r, iris_rw};
    task.kernel(name, 1, nullptr, gws, lws, 4, params, pinfo);
  } else {
    void* params[5] = {&nodes, &mem_nidx, &mem_nlist, &mem_nstat, mem_wl};
    int pinfo[5] = {sizeof(nodes), iris_r, iris_r, iris_rw, iris_rw};
    task.kernel(name, 1, nullptr, gws, lws, 5, params, pinfo);
  }
  task.submit(iris_gpu, nullptr, 1);
}

static void computeCC(const int repeat,
                      const int nodes,
                      const int edges,
                      const int* const nidx,
                      const int* const nlist,
                      int* const nstat)
{
  const int blocks = 64;
  std::printf("Number of multiprocessor = IRIS-HIP\n");
  std::printf("Max threads per multiprocessor = IRIS-HIP\n");
  std::printf("Number of thread blocks in a grid = %d\n", blocks);

  int* wl = static_cast<int*>(std::malloc(nodes * sizeof(int)));
  if (wl == nullptr) {
    std::fprintf(stderr, "ERROR: could not allocate worklist\n\n");
    std::exit(1);
  }

  iris::DMem mem_nidx(const_cast<int*>(nidx), (nodes + 1) * sizeof(int));
  iris::DMem mem_nlist(const_cast<int*>(nlist), edges * sizeof(int));
  iris::DMem mem_nstat(nstat, nodes * sizeof(int));
  iris::DMem mem_wl(wl, nodes * sizeof(int));

  auto start = std::chrono::high_resolution_clock::now();
  for (int n = 0; n < repeat; n++) {
    launch_kernel("cc_init", nodes, mem_nidx, mem_nlist, mem_nstat, nullptr, blocks);
    launch_kernel("cc_compute1", nodes, mem_nidx, mem_nlist, mem_nstat, &mem_wl, blocks);
    launch_kernel("cc_compute2", nodes, mem_nidx, mem_nlist, mem_nstat, &mem_wl, blocks);
    launch_kernel("cc_compute3", nodes, mem_nidx, mem_nlist, mem_nstat, &mem_wl, blocks);
    launch_kernel("cc_flatten", nodes, mem_nidx, mem_nlist, mem_nstat, nullptr, blocks);
  }

  iris::Task flush;
  flush.flush_out(mem_nstat);
  flush.submit(iris_gpu, nullptr, 1);
  mem_nstat.fetch(nstat);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  float runtime = static_cast<float>(elapsed_seconds.count() / repeat);

  std::printf("Average kernel execution time: %.4f s\n", runtime);
  std::printf("throughput: %.3f Mnodes/s\n", nodes * 0.000001f / runtime);
  std::printf("throughput: %.3f Medges/s\n", edges * 0.000001f / runtime);
  std::free(wl);
}

static void verify(const int v,
                   const int id,
                   const int* const nidx,
                   const int* const nlist,
                   int* const nstat,
                   bool& ok)
{
  if (nstat[v] >= 0) {
    if (nstat[v] != id) {
      std::fprintf(stderr, "ERROR: found incorrect ID value\n\n");
      ok = false;
      return;
    }
    nstat[v] = -1;
    for (int i = nidx[v]; i < nidx[v + 1]; i++) {
      verify(nlist[i], id, nidx, nlist, nstat, ok);
    }
  }
}

int main(int argc, char* argv[])
{
  std::printf("ECL-CC v1.1 (%s)\n", __FILE__);
  std::printf("Copyright 2017-2020 Texas State University\n");

  if (argc != 3) {
    std::fprintf(stderr, "USAGE: %s <input_file_name> <repeat>\n\n", argv[0]);
    return 1;
  }

  ECLgraph g = readECLgraph(argv[1]);
  const int repeat = std::atoi(argv[2]);

  int* nodestatus = static_cast<int*>(std::malloc(g.nodes * sizeof(int)));
  if (nodestatus == nullptr) {
    std::fprintf(stderr, "ERROR: nodestatus - host memory allocation failed\n\n");
    freeECLgraph(g);
    return 1;
  }

  std::printf("input graph: %d nodes and %d edges (%s)\n", g.nodes, g.edges, argv[1]);
  std::printf("average degree: %.2f edges per node\n", 1.0 * g.edges / g.nodes);
  int mindeg = g.nodes;
  int maxdeg = 0;
  for (int v = 0; v < g.nodes; v++) {
    int deg = g.nindex[v + 1] - g.nindex[v];
    mindeg = std::min(mindeg, deg);
    maxdeg = std::max(maxdeg, deg);
  }
  std::printf("minimum degree: %d edges\n", mindeg);
  std::printf("maximum degree: %d edges\n", maxdeg);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  computeCC(repeat, g.nodes, g.edges, g.nindex, g.nlist, nodestatus);
  platform.finalize();

  std::set<int> s1;
  for (int v = 0; v < g.nodes; v++) {
    s1.insert(nodestatus[v]);
  }
  std::printf("number of connected components: %zu\n", s1.size());

  bool ok = true;
  for (int v = 0; v < g.nodes; v++) {
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      if (nodestatus[g.nlist[i]] != nodestatus[v]) {
        std::fprintf(stderr, "ERROR: found adjacent nodes in different components\n\n");
        ok = false;
        break;
      }
    }
  }

  for (int v = 0; v < g.nodes; v++) {
    if (nodestatus[v] < 0) {
      std::fprintf(stderr, "ERROR: found negative component number\n\n");
      ok = false;
      break;
    }
  }

  std::set<int> s2;
  size_t count = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (nodestatus[v] >= 0) {
      count++;
      s2.insert(nodestatus[v]);
      verify(v, nodestatus[v], g.nindex, g.nlist, nodestatus, ok);
    }
  }
  if (s1.size() != s2.size()) {
    std::fprintf(stderr, "ERROR: number of components do not match\n\n");
    ok = false;
  }

  if (s1.size() != count) {
    std::fprintf(stderr, "ERROR: component IDs are not unique\n\n");
    ok = false;
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  std::free(nodestatus);
  freeECLgraph(g);
  return ok ? 0 : 1;
}
