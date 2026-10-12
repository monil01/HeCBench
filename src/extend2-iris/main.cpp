#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "read_data.h"

static void check(int a, int b, const char* s, int* errors)
{
  if (a != b) {
    printf("Error: %s %d %d\n", s, a, b);
    (*errors)++;
  }
}

typedef struct {
  int h, e;
} eh_t;

static float extend2(struct extend2_dat* d, int* errors)
{
  eh_t* eh = nullptr;
  char* qp = nullptr;
  posix_memalign((void**)&eh, 64, (d->qlen + 1) * sizeof(eh_t));
  posix_memalign((void**)&qp, 64, d->qlen * d->m);
  memset(eh, 0, (d->qlen + 1) * sizeof(eh_t));
  memset(qp, 0, d->qlen * d->m);

  int qle = 0, tle = 0, gtle = 0, gscore = 0, max_off = 0, score = 0;

  int qlen = d->qlen;
  int tlen = d->tlen;
  int m = d->m;
  int o_del = d->o_del;
  int e_del = d->e_del;
  int o_ins = d->o_ins;
  int e_ins = d->e_ins;
  int w = d->w;
  int end_bonus = d->end_bonus;
  int zdrop = d->zdrop;
  int h0 = d->h0;

  auto start = std::chrono::steady_clock::now();

  iris::DMem d_query(d->query, qlen);
  iris::DMem d_target(d->target, tlen);
  iris::DMem d_mat(d->mat, m * m);
  iris::DMem d_eh(eh, (qlen + 1) * sizeof(eh_t));
  iris::DMem d_qp(qp, qlen * m);
  iris::DMem d_qle(&qle, sizeof(int));
  iris::DMem d_tle(&tle, sizeof(int));
  iris::DMem d_gtle(&gtle, sizeof(int));
  iris::DMem d_gscore(&gscore, sizeof(int));
  iris::DMem d_max_off(&max_off, sizeof(int));
  iris::DMem d_score(&score, sizeof(int));

  size_t gws[1] = {1};
  size_t lws[1] = {1};
  void* params[22] = {&d_query, &d_target, &d_mat, &d_eh, &d_qp,
                      &d_qle, &d_tle, &d_gtle, &d_gscore, &d_max_off,
                      &d_score, &qlen, &tlen, &m, &o_del, &e_del,
                      &o_ins, &e_ins, &w, &end_bonus, &zdrop, &h0};
  int pinfo[22] = {iris_r, iris_r, iris_r, iris_rw, iris_rw,
                   iris_w, iris_w, iris_w, iris_w, iris_w,
                   iris_w, sizeof(qlen), sizeof(tlen), sizeof(m), sizeof(o_del),
                   sizeof(e_del), sizeof(o_ins), sizeof(e_ins), sizeof(w),
                   sizeof(end_bonus), sizeof(zdrop), sizeof(h0)};
  iris::Task task;
  task.kernel("kernel_extend2", 1, nullptr, gws, lws, 22, params, pinfo);
  task.flush_out(d_qle);
  task.flush_out(d_tle);
  task.flush_out(d_gtle);
  task.flush_out(d_gscore);
  task.flush_out(d_max_off);
  task.flush_out(d_score);
  task.submit(iris_gpu, nullptr, 1);
  d_qle.fetch(&qle);
  d_tle.fetch(&tle);
  d_gtle.fetch(&gtle);
  d_gscore.fetch(&gscore);
  d_max_off.fetch(&max_off);
  d_score.fetch(&score);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  check(d->qle, qle, "qle", errors);
  check(d->tle, tle, "tle", errors);
  check(d->gtle, gtle, "gtle", errors);
  check(d->gscore, gscore, "gscore", errors);
  check(d->max_off, max_off, "max_off", errors);
  check(d->score, score, "score", errors);

  free(eh);
  free(qp);
  return static_cast<float>(time);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int repeat = atoi(argv[1]);
  struct extend2_dat d;
  const char* files[] = {
#include "filelist.txt"
  };

  int errors = 0;
  float time = 0.f;
  for (int f = 0; f < repeat; f++) {
    if (!read_data(files[f % 17], &d)) {
      platform.finalize();
      return 1;
    }
    time += extend2(&d, &errors);
    free(d.query);
    free(d.target);
  }

  printf("Average offload time %f (us)\n", (time * 1e-3f) / repeat);
  printf("%s\n", errors == 0 ? "PASS" : "FAIL");

  platform.finalize();
  return errors == 0 ? 0 : 1;
}
