#include "curvilinear4sg.h"

void curvilinear4sg_ci(
    int ifirst, int ilast,
    int jfirst, int jlast,
    int kfirst, int klast,
    iris::DMem &d_u,
    iris::DMem &d_mu,
    iris::DMem &d_lambda,
    iris::DMem &d_met,
    iris::DMem &d_jac,
    iris::DMem &d_lu,
    int* onesided,
    iris::DMem &d_acof,
    iris::DMem &d_bope,
    iris::DMem &d_ghcof,
    iris::DMem &d_acof_no_gp,
    iris::DMem &d_ghcof_no_gp,
    iris::DMem &d_strx,
    iris::DMem &d_stry,
    int nk, char op) {

  iris::Task task;
  float_sw4 a1 = 0;
  float_sw4 sgn = 1;
  if (op == '=') {
    a1 = 0;
    sgn = 1;
  } else if (op == '+') {
    a1 = 1;
    sgn = 1;
  } else if (op == '-') {
    a1 = 1;
    sgn = -1;
  }

  int kstart = kfirst + 2;
  int kend = klast - 2;
  if (onesided[5] == 1) kend = nk - 6;

  if (onesided[4] == 1) {
    kstart = 7;

    Range<16> I(ifirst + 2, ilast - 1);
    Range<4> J(jfirst + 2, jlast - 1);
    Range<3> K(1, 6 + 1);  // This was 6

    {
    void *args[] = { &I.start, &I.end, &J.start, &J.end, &K.start, &K.end, &ifirst, &ilast, &jfirst, &jlast, &kfirst, &klast, &a1, &sgn, &d_u, &d_mu, &d_lambda, &d_met, &d_jac, &d_lu, &d_acof, &d_bope, &d_ghcof, &d_acof_no_gp, &d_ghcof_no_gp, &d_strx, &d_stry };
    int info[] = { sizeof(I.start), sizeof(I.end), sizeof(J.start), sizeof(J.end), sizeof(K.start), sizeof(K.end), sizeof(ifirst), sizeof(ilast), sizeof(jfirst), sizeof(jlast), sizeof(kfirst), sizeof(klast), sizeof(a1), sizeof(sgn), iris_r, iris_r, iris_r, iris_r, iris_r, iris_rw, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r };
    size_t gws[] = { size_t(I.blocks * I.tpb), size_t(J.blocks * J.tpb), size_t(K.blocks * K.tpb) };
    size_t lws[] = { size_t(I.tpb), size_t(J.tpb), size_t(K.tpb) };
    iris_check(task.kernel("kernel1", 3, nullptr, gws, lws, 27, args, info));
    }
  }

  Range<64> I(ifirst + 2, ilast - 1);
  Range<2> J(jfirst + 2, jlast - 1);
  Range<2> K(kstart, kend + 1);  // Changed for CUrvi-MR Was klast-1

  {
    void *args[] = { &I.start, &I.end, &J.start, &J.end, &K.start, &K.end, &ifirst, &ilast, &jfirst, &jlast, &kfirst, &klast, &a1, &sgn, &d_u, &d_mu, &d_lambda, &d_met, &d_jac, &d_lu, &d_acof, &d_bope, &d_ghcof, &d_acof_no_gp, &d_ghcof_no_gp, &d_strx, &d_stry };
    int info[] = { sizeof(I.start), sizeof(I.end), sizeof(J.start), sizeof(J.end), sizeof(K.start), sizeof(K.end), sizeof(ifirst), sizeof(ilast), sizeof(jfirst), sizeof(jlast), sizeof(kfirst), sizeof(klast), sizeof(a1), sizeof(sgn), iris_r, iris_r, iris_r, iris_r, iris_r, iris_rw, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r };
    size_t gws[] = { size_t(I.blocks * I.tpb), size_t(J.blocks * J.tpb), size_t(K.blocks * K.tpb) };
    size_t lws[] = { size_t(I.tpb), size_t(J.tpb), size_t(K.tpb) };
    iris_check(task.kernel("kernel2", 3, nullptr, gws, lws, 27, args, info));
    }

  {
    void *args[] = { &I.start, &I.end, &J.start, &J.end, &K.start, &K.end, &ifirst, &ilast, &jfirst, &jlast, &kfirst, &klast, &a1, &sgn, &d_u, &d_mu, &d_lambda, &d_met, &d_jac, &d_lu, &d_acof, &d_bope, &d_ghcof, &d_acof_no_gp, &d_ghcof_no_gp, &d_strx, &d_stry };
    int info[] = { sizeof(I.start), sizeof(I.end), sizeof(J.start), sizeof(J.end), sizeof(K.start), sizeof(K.end), sizeof(ifirst), sizeof(ilast), sizeof(jfirst), sizeof(jlast), sizeof(kfirst), sizeof(klast), sizeof(a1), sizeof(sgn), iris_r, iris_r, iris_r, iris_r, iris_r, iris_rw, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r };
    size_t gws[] = { size_t(I.blocks * I.tpb), size_t(J.blocks * J.tpb), size_t(K.blocks * K.tpb) };
    size_t lws[] = { size_t(I.tpb), size_t(J.tpb), size_t(K.tpb) };
    iris_check(task.kernel("kernel3", 3, nullptr, gws, lws, 27, args, info));
    }

  {
    void *args[] = { &I.start, &I.end, &J.start, &J.end, &K.start, &K.end, &ifirst, &ilast, &jfirst, &jlast, &kfirst, &klast, &a1, &sgn, &d_u, &d_mu, &d_lambda, &d_met, &d_jac, &d_lu, &d_acof, &d_bope, &d_ghcof, &d_acof_no_gp, &d_ghcof_no_gp, &d_strx, &d_stry };
    int info[] = { sizeof(I.start), sizeof(I.end), sizeof(J.start), sizeof(J.end), sizeof(K.start), sizeof(K.end), sizeof(ifirst), sizeof(ilast), sizeof(jfirst), sizeof(jlast), sizeof(kfirst), sizeof(klast), sizeof(a1), sizeof(sgn), iris_r, iris_r, iris_r, iris_r, iris_r, iris_rw, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r };
    size_t gws[] = { size_t(I.blocks * I.tpb), size_t(J.blocks * J.tpb), size_t(K.blocks * K.tpb) };
    size_t lws[] = { size_t(I.tpb), size_t(J.tpb), size_t(K.tpb) };
    iris_check(task.kernel("kernel4", 3, nullptr, gws, lws, 27, args, info));
    }


  if (onesided[5] == 1) {
    Range<16> I(ifirst + 2, ilast - 1);
    Range<4> J(jfirst + 2, jlast - 1);
    Range<4> K(nk - 5, nk + 1);  // THIS WAS 6

    {
    void *args[] = { &I.start, &I.end, &J.start, &J.end, &K.start, &K.end, &ifirst, &ilast, &jfirst, &jlast, &kfirst, &klast, &nk, &a1, &sgn, &d_u, &d_mu, &d_lambda, &d_met, &d_jac, &d_lu, &d_acof, &d_bope, &d_ghcof, &d_acof_no_gp, &d_ghcof_no_gp, &d_strx, &d_stry };
    int info[] = { sizeof(I.start), sizeof(I.end), sizeof(J.start), sizeof(J.end), sizeof(K.start), sizeof(K.end), sizeof(ifirst), sizeof(ilast), sizeof(jfirst), sizeof(jlast), sizeof(kfirst), sizeof(klast), sizeof(nk), sizeof(a1), sizeof(sgn), iris_r, iris_r, iris_r, iris_r, iris_r, iris_rw, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r };
    size_t gws[] = { size_t(I.blocks * I.tpb), size_t(J.blocks * J.tpb), size_t(K.blocks * K.tpb) };
    size_t lws[] = { size_t(I.tpb), size_t(J.tpb), size_t(K.tpb) };
    iris_check(task.kernel("kernel5", 3, nullptr, gws, lws, 28, args, info));
    }
  }
  iris_check(task.submit(iris_gpu, nullptr, true));
}
