#include <hip/hip_runtime.h>
#include <stdlib.h>

typedef struct {
  int h, e;
} eh_t;

extern "C" __global__ void kernel_extend2(const unsigned char* query,
                                           const unsigned char* target,
                                           const char* mat,
                                           eh_t* eh,
                                           char* qp,
                                           int* qle_acc,
                                           int* tle_acc,
                                           int* gtle_acc,
                                           int* gscore_acc,
                                           int* max_off_acc,
                                           int* score_acc,
                                           const int qlen,
                                           const int tlen,
                                           const int m,
                                           const int o_del,
                                           const int e_del,
                                           const int o_ins,
                                           const int e_ins,
                                           int w,
                                           const int end_bonus,
                                           const int zdrop,
                                           const int h0)
{
  int oe_del = o_del + e_del;
  int oe_ins = o_ins + e_ins;
  int i, j, k;
  int beg, end;
  int max, max_i, max_j, max_ins, max_del, max_ie;
  int gscore;
  int max_off;

  for (k = i = 0; k < m; ++k) {
    const char* p = mat + k * m;
    for (j = 0; j < qlen; ++j) {
      qp[i++] = p[query[j]];
    }
  }

  eh[0].h = h0;
  eh[1].h = h0 > oe_ins ? h0 - oe_ins : 0;
  for (j = 2; j <= qlen && eh[j - 1].h > e_ins; ++j) {
    eh[j].h = eh[j - 1].h - e_ins;
  }

  k = m * m;
  for (i = 0, max = 0; i < k; ++i) {
    max = max > mat[i] ? max : mat[i];
  }
  max_ins = (int)((float)(qlen * max + end_bonus - o_ins) / e_ins + 1.f);
  max_ins = max_ins > 1 ? max_ins : 1;
  w = w < max_ins ? w : max_ins;
  max_del = (int)((float)(qlen * max + end_bonus - o_del) / e_del + 1.f);
  max_del = max_del > 1 ? max_del : 1;
  w = w < max_del ? w : max_del;

  max = h0;
  max_i = max_j = -1;
  max_ie = -1;
  gscore = -1;
  max_off = 0;
  beg = 0;
  end = qlen;
  for (i = 0; i < tlen; ++i) {
    int t, f = 0, h1, mm = 0, mj = -1;
    char* q = qp + target[i] * qlen;

    if (beg < i - w) beg = i - w;
    if (end > i + w + 1) end = i + w + 1;
    if (end > qlen) end = qlen;

    if (beg == 0) {
      h1 = h0 - (o_del + e_del * (i + 1));
      if (h1 < 0) h1 = 0;
    } else {
      h1 = 0;
    }

    for (j = beg; j < end; ++j) {
      eh_t* p = eh + j;
      int h, M = p->h, e = p->e;
      p->h = h1;
      M = M ? M + q[j] : 0;
      h = M > e ? M : e;
      h = h > f ? h : f;
      h1 = h;
      mj = mm > h ? mj : j;
      mm = mm > h ? mm : h;
      t = M - oe_del;
      t = t > 0 ? t : 0;
      e -= e_del;
      e = e > t ? e : t;
      p->e = e;
      t = M - oe_ins;
      t = t > 0 ? t : 0;
      f -= e_ins;
      f = f > t ? f : t;
    }
    eh[end].h = h1;
    eh[end].e = 0;
    if (j == qlen) {
      max_ie = gscore > h1 ? max_ie : i;
      gscore = gscore > h1 ? gscore : h1;
    }
    if (mm == 0) break;
    if (mm > max) {
      max = mm;
      max_i = i;
      max_j = mj;
      max_off = max_off > abs(mj - i) ? max_off : abs(mj - i);
    } else if (zdrop > 0) {
      if (i - max_i > mj - max_j) {
        if (max - mm - ((i - max_i) - (mj - max_j)) * e_del > zdrop) break;
      } else {
        if (max - mm - ((mj - max_j) - (i - max_i)) * e_ins > zdrop) break;
      }
    }
    for (j = beg; j < end && eh[j].h == 0 && eh[j].e == 0; ++j) {
    }
    beg = j;
    for (j = end; j >= beg && eh[j].h == 0 && eh[j].e == 0; --j) {
    }
    end = j + 2 < qlen ? j + 2 : qlen;
  }
  *qle_acc = max_j + 1;
  *tle_acc = max_i + 1;
  *gtle_acc = max_ie + 1;
  *gscore_acc = gscore;
  *max_off_acc = max_off;
  *score_acc = max;
}
