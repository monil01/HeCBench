#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <iris/iris.hpp>
#include "utils.h"
#include "evidence.h"

static void check(int status) {
  if (status != IRIS_SUCCESS) { fprintf(stderr, "IRIS operation failed: %d\n", status); exit(1); }
}

static void computeRSMT(const int* idxin, const coord_t* xin, const coord_t* yin,
                       int* idxout, coord_t* xout, coord_t* yout, edge* edges, int numnets) {
  iris::Platform platform;
  check(platform.init(nullptr, nullptr, true));
  int device = -1;
  for (int d = 0; d < platform.ndevs(); ++d) {
    int backend = -1;
    check(iris_device_info(d, iris_backend, &backend, nullptr));
    if (backend == iris_hip) { device = d; break; }
  }
  if (device < 0) { fprintf(stderr, "No IRIS HIP device\n"); exit(1); }
  char name[256] = {};
  check(iris_device_info(device, iris_name, name, nullptr));
  int wave = 0;
  {
    iris::DMem info(&wave, sizeof(wave));
    iris::Task probe;
    void* p[] = {&info}; int modes[] = {iris_w};
    size_t gws = 64, lws = 64;
    check(probe.kernel("waveInfo", 1, nullptr, &gws, &lws, 1, p, modes));
    check(probe.flush_out(info));
    check(probe.submit(device, nullptr, true));
  }
  if (wave != 64) { fprintf(stderr, "This HIP kernel requires wave64; got %d\n", wave); exit(1); }
  const int blocks = 240; // MI100: 120 compute units, same 2x launch as native HIP.
  printf("IRIS backend: HIP; device: %s; wave size: %d\n", name, wave);
  printf("launching %d thread blocks with %d threads per block (warp size %d)\n", blocks, 768, wave);
  const int size = idxin[numnets];
  std::fill(xout, xout + 2 * size, -1);
  std::fill(yout, yout + 2 * size, -1);
  std::fill(edges, edges + 2 * size, edge{0, 0});
  std::vector<int> wl(numnets, 0);
  int counters[3] = {0, 0, 0};
  auto start = std::chrono::steady_clock::now();
  {
    iris::DMem mi((void*)idxin, (numnets+1)*sizeof(int));
    iris::DMem mx((void*)xin, size*sizeof(coord_t)), my((void*)yin, size*sizeof(coord_t));
    iris::DMem mo(idxout, (numnets+1)*sizeof(int));
    iris::DMem ox(xout, 2*size*sizeof(coord_t)), oy(yout, 2*size*sizeof(coord_t));
    iris::DMem me(edges, 2*size*sizeof(edge)), mw(wl.data(), numnets*sizeof(int));
    iris::DMem mc(counters, sizeof(counters));
    iris::Task small;
    void* p[] = {&mi,&mx,&my,&mo,&ox,&oy,&me,&numnets,&mw,&mc};
    int modes[] = {iris_r,iris_r,iris_r,iris_w,iris_rw,iris_rw,iris_rw,sizeof(numnets),iris_rw,iris_rw};
    size_t gws = blocks*768, lws = 768;
    check(small.kernel("largeNetKernel", 1, nullptr, &gws, &lws, 10, p, modes));
    check(small.submit(device, nullptr, true));
    iris::Task large;
    void* q[] = {&mi,&ox,&oy,&me,&mw,&mc};
    int qm[] = {iris_r,iris_rw,iris_rw,iris_rw,iris_r,iris_rw};
    gws = blocks*192; lws = 192;
    check(large.kernel("smallNetKernel", 1, nullptr, &gws, &lws, 6, q, qm));
    check(large.flush_out(mo));
    check(large.flush_out(ox));
    check(large.flush_out(oy));
    check(large.flush_out(me));
    check(large.flush_out(mc));
    check(large.submit(device, nullptr, true));
  }
  auto end = std::chrono::steady_clock::now();
  double runtime = std::chrono::duration<double>(end-start).count();
  printf("compute time: %.6f s\n", runtime);
  printf("throughput: %.f nets/sec\n", numnets/runtime);
  if (platform.error_count()) { fprintf(stderr, "IRIS runtime errors\n"); exit(1); }
  check(platform.finalize());
}

int main(int argc, char* argv[])
{
  printf("A Simple, Fast, and GPU-friendly Steiner-Tree Heuristic\n");
  printf("Copyright 2019-2022 Texas State University\n\n");

  if (argc != 2) {
    printf("Usage: %s file_name \n", argv[0]);
    exit(-1);
  }

  printf("reading input file: %s\n", argv[1]);
  grid g;
  net_list n;
  read_file(argv[1], g, n);
  const int numnets = n.num_net;

  int* idxin = NULL;
  idxin = static_cast<decltype(idxin)>(malloc((numnets + 1) * sizeof(int)));
  if (!idxin) { fprintf(stderr, "host allocation failed\n"); return 1; }

  // initialize idxin
  idxin[0] = 0;
  ID hipin = 0;
  int pos = 0;
  for (int i = 0; i < numnets; i++) {
    const ID num = std::min(n.net_num_pins[i], MaxPins);
    hipin = std::max(hipin, num);
    pos += num;
    idxin[i + 1] = pos;
  }

  // histogram and trucated nets
  int trunc = 0;
  int* const hist = new int [hipin + 1];
  for (int i = 0; i < hipin + 1; i++) hist[i] = 0;
  for (int i = 0; i < numnets; i++) {
    const ID num = std::min(n.net_num_pins[i], MaxPins);
    if (num < n.net_num_pins[i]) trunc++;
    hist[num]++;
  }
  int sum = 0;
  for (int i = 0; i < hipin + 1; i++) {
    sum += hist[i];
  }
  delete [] hist;

  printf("number of nets: %d\n", numnets);
  printf("max pins per net: %d\n", hipin);
  printf("truncated nets: %d\n", trunc);
  if (hipin > MaxPins) {printf("ERROR: hi_pin_count must be no more than %d\n", MaxPins); exit(-1);}

  // pin coordinates
  coord_t* xin = NULL;
  xin = static_cast<decltype(xin)>(malloc(idxin[numnets] * sizeof(coord_t)));
  if (!xin) { fprintf(stderr, "host allocation failed\n"); return 1; }

  coord_t* yin = NULL;
  yin = static_cast<decltype(yin)>(malloc(idxin[numnets] * sizeof(coord_t)));
  if (!yin) { fprintf(stderr, "host allocation failed\n"); return 1; }

  // initialize pin coordinates
  pos = 0;
  for (int i = 0; i < numnets; i++) {
    const ID num = idxin[i + 1] - idxin[i];
    for (ID j = 0; j < num; j++) xin[pos + j] = std::get<0>(n.num_net_arr[i][j]);
    for (ID j = 0; j < num; j++) yin[pos + j] = std::get<1>(n.num_net_arr[i][j]);
    pos += num;
  }

  // result storage
  const int size = 2 * idxin[numnets];
  int* idxout = NULL;
  idxout = static_cast<decltype(idxout)>(malloc((numnets + 1) * sizeof(int)));
  if (!idxout) { fprintf(stderr, "host allocation failed\n"); return 1; }

  coord_t* xout = NULL;
  xout = static_cast<decltype(xout)>(malloc(size * sizeof(coord_t)));
  if (!xout) { fprintf(stderr, "host allocation failed\n"); return 1; }

  coord_t* yout = NULL;
  yout = static_cast<decltype(yout)>(malloc(size * sizeof(coord_t)));
  if (!yout) { fprintf(stderr, "host allocation failed\n"); return 1; }

  edge* edges = NULL;
  edges = static_cast<decltype(edges)>(malloc(size * sizeof(edge)));
  if (!edges) { fprintf(stderr, "host allocation failed\n"); return 1; }

  // compute Steiner points and edges
  computeRSMT(idxin, xin, yin, idxout, xout, yout, edges, numnets);

  dumpResults(idxin, idxout, xout, yout, edges, numnets);

  // print results
  long total_len = 0, total_pin = 0;

  for (int i = 0; i < numnets; i++) {
    // body of treeLength function illustrates how to read solution
    const coord_t len = treeLength(idxout[i + 1] - idxout[i],
                      &xout[idxout[i]], &yout[idxout[i]], &edges[idxout[i]]);
    const ID num = std::min(n.net_num_pins[i], MaxPins);
    total_len += len;
    total_pin += num;
  }
  printf("total wirelength: %ld\n", total_len);
  printf("total pins: %ld\n", total_pin);

  // clean up
  free_memory(g, n);
  free(edges);
  free(yout);
  free(xout);
  free(idxout);
  free(yin);
  free(xin);
  free(idxin);

  return 0;
}
