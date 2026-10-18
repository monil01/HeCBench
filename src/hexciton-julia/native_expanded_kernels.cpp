__global__
void comm_empty(
    real_2_t *__restrict__ sigma_in,
    real_2_t *__restrict__ sigma_out,
    real_2_t *__restrict__ hamiltonian)
{
}
__global__
void comm_init (
    const real_2_t *__restrict__ sigma_in,
          real_2_t *__restrict__ sigma_out,
    const real_2_t *__restrict__ hamiltonian,
    const int dim)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int sigma_id = gid * dim * dim;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      real_2_t tmp;
      tmp.x = 0.0;
      tmp.y = 0.0;
      for (int k = 0; k < dim; ++k) {
        tmp.x += (hamiltonian[i * dim + k].x * sigma_in[sigma_id + k * dim + j].x -
            sigma_in[sigma_id + i * dim + k].x * hamiltonian[k * dim + j].x);
        tmp.x -= (hamiltonian[i * dim + k].y * sigma_in[sigma_id + k * dim + j].y -
            sigma_in[sigma_id + i * dim + k].y * hamiltonian[k * dim + j].y);
        tmp.y += (hamiltonian[i * dim + k].x * sigma_in[sigma_id + k * dim + j].y -
            sigma_in[sigma_id + i * dim + k].x * hamiltonian[k * dim + j].y);
        tmp.y += (hamiltonian[i * dim + k].y * sigma_in[sigma_id + k * dim + j].x -
            sigma_in[sigma_id + i * dim + k].y * hamiltonian[k * dim + j].x);
      }
      sigma_out[sigma_id + i * dim + j].x += hdt * tmp.y;
      sigma_out[sigma_id + i * dim + j].y -= hdt * tmp.x;
    }
  }
}
__global__
void comm_refactor(
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int sigma_id = gid * dim * dim * 2;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      real_t tmp_real = 0.0;
      real_t tmp_imag = 0.0;
      for (int k = 0; k < dim; ++k) {
        tmp_real += hamiltonian[(2 * ((i) * dim + (k)))] * sigma_in[(sigma_id + 2 * ((k) * dim + (j)))];
        tmp_real -= sigma_in[(sigma_id + 2 * ((i) * dim + (k)))] * hamiltonian[(2 * ((k) * dim + (j)))];
        tmp_real -= hamiltonian[(2 * ((i) * dim + (k)) + 1)] * sigma_in[(sigma_id + 2 * ((k) * dim + (j)) + 1)];
        tmp_real += sigma_in[(sigma_id + 2 * ((i) * dim + (k)) + 1)] * hamiltonian[(2 * ((k) * dim + (k)) + 1)];
        tmp_imag += hamiltonian[(2 * ((i) * dim + (k)))] * sigma_in[(sigma_id + 2 * ((k) * dim + (j)) + 1)];
        tmp_imag -= sigma_in[(sigma_id + 2 * ((i) * dim + (k)))] * hamiltonian[(2 * ((k) * dim + (k)) + 1)];
        tmp_imag += hamiltonian[(2 * ((i) * dim + (k)) + 1)] * sigma_in[(sigma_id + 2 * ((k) * dim + (j)))];
        tmp_imag -= sigma_in[(sigma_id + 2 * ((i) * dim + (k)) + 1)] * hamiltonian[(2 * ((k) * dim + (j)))];
      }
      sigma_out[(sigma_id + 2 * ((i) * dim + (j)))] += hdt * tmp_imag;
      sigma_out[(sigma_id + 2 * ((i) * dim + (j)) + 1)] -= hdt * tmp_real;
    }
  }
}
__global__
void comm_refactor_direct_store(
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int sigma_id = gid * dim * dim * 2;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        sigma_out[(sigma_id + 2 * ((i) * dim + (j)))] += hamiltonian[(2 * ((i) * dim + (k)))] * sigma_in[(sigma_id + 2 * ((k) * dim + (j)) + 1)];
        sigma_out[(sigma_id + 2 * ((i) * dim + (j)))] -= sigma_in[(sigma_id + 2 * ((i) * dim + (k)))] * hamiltonian[(2 * ((k) * dim + (k)) + 1)];
        sigma_out[(sigma_id + 2 * ((i) * dim + (j)))] += hamiltonian[(2 * ((i) * dim + (k)) + 1)] * sigma_in[(sigma_id + 2 * ((k) * dim + (j)))];
        sigma_out[(sigma_id + 2 * ((i) * dim + (j)))] -= sigma_in[(sigma_id + 2 * ((i) * dim + (k)) + 1)] * hamiltonian[(2 * ((k) * dim + (j)))];
        sigma_out[(sigma_id + 2 * ((i) * dim + (j)) + 1)] -= hamiltonian[(2 * ((i) * dim + (k)))] * sigma_in[(sigma_id + 2 * ((k) * dim + (j)))];
        sigma_out[(sigma_id + 2 * ((i) * dim + (j)) + 1)] += sigma_in[(sigma_id + 2 * ((i) * dim + (k)))] * hamiltonian[(2 * ((k) * dim + (j)))];
        sigma_out[(sigma_id + 2 * ((i) * dim + (j)) + 1)] += hamiltonian[(2 * ((i) * dim + (k)) + 1)] * sigma_in[(sigma_id + 2 * ((k) * dim + (j)) + 1)];
        sigma_out[(sigma_id + 2 * ((i) * dim + (j)) + 1)] -= sigma_in[(sigma_id + 2 * ((i) * dim + (k)) + 1)] * hamiltonian[(2 * ((k) * dim + (k)) + 1)];
      }
    }
  }
}
__global__
void comm_aosoa_naive(
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      real_t tmp_real = 0.0;
      real_t tmp_imag = 0.0;
      for (int k = 0; k < dim; ++k) {
        tmp_imag -= hamiltonian[((i) * dim + (k))] * sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + ((gid % 4)))];
        tmp_imag += sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + ((gid % 4)))] * hamiltonian[((k) * dim + (j))];
        tmp_imag += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + 4 + ((gid % 4)))];
        tmp_imag -= sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + 4 + ((gid % 4)))] * hamiltonian[(dim * dim + (k) * dim + (j))];
        tmp_real += hamiltonian[((i) * dim + (k))] * sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + 4 + ((gid % 4)))];
        tmp_real -= sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + ((gid % 4)))] * hamiltonian[(dim * dim + (k) * dim + (j))];
        tmp_real += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + ((gid % 4)))];
        tmp_real -= sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + 4 + ((gid % 4)))] * hamiltonian[((k) * dim + (j))];
      }
      sigma_out[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + ((gid % 4)))] += tmp_real;
      sigma_out[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + 4 + ((gid % 4)))] += tmp_imag;
    }
  }
}
__global__
void comm_aosoa_naive_constants (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      real_t tmp_real = 0.0;
      real_t tmp_imag = 0.0;
      for (int k = 0; k < 7; ++k) {
        tmp_imag -= hamiltonian[((i) * 7 + (k))] * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4)))];
        tmp_imag += sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4)))] * hamiltonian[((k) * 7 + (j))];
        tmp_imag += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4)))];
        tmp_imag -= sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4)))] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += hamiltonian[((i) * 7 + (k))] * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4)))];
        tmp_real -= sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4)))] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4)))];
        tmp_real -= sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4)))] * hamiltonian[((k) * 7 + (j))];
      }
      sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))] += tmp_real;
      sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))] += tmp_imag;
    }
  }
}
__global__
void comm_aosoa_naive_constants_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 7; ++i) {
    for (int k = 0; k < 7; ++k) {
      real_t ham_real_tmp = hamiltonian[((i) * 7 + (k))];
      real_t ham_imag_tmp = hamiltonian[(7 * 7 + (i) * 7 + (k))];
      real_t sigma_real_tmp = sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4)))];
      real_t sigma_imag_tmp = sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4)))];
      for (int j = 0; j < 7; ++j) {
        real_t tmp_real = sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))];
        real_t tmp_imag = sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))];
        tmp_imag -= ham_real_tmp * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4)))];
        tmp_imag += sigma_real_tmp * hamiltonian[((k) * 7 + (j))];
        tmp_imag += ham_imag_tmp * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4)))];
        tmp_imag -= sigma_imag_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += ham_real_tmp * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4)))];
        tmp_real -= sigma_real_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += ham_imag_tmp * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4)))];
        tmp_real -= sigma_imag_tmp * hamiltonian[((k) * 7 + (j))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))] = tmp_real;
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))] = tmp_imag;
      }
    }
  }
}
__global__
void comm_aosoa_naive_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        sigma_out[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + 4 + ((gid % 4)))] -= hamiltonian[((i) * dim + (k))] * sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + 4 + ((gid % 4)))] += sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + ((gid % 4)))] * hamiltonian[((k) * dim + (j))];
        sigma_out[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + 4 + ((gid % 4)))] += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + 4 + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + 4 + ((gid % 4)))] -= sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + 4 + ((gid % 4)))] * hamiltonian[(dim * dim + (k) * dim + (j))];
        sigma_out[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + ((gid % 4)))] += hamiltonian[((i) * dim + (k))] * sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + 4 + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + ((gid % 4)))] -= sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + ((gid % 4)))] * hamiltonian[(dim * dim + (k) * dim + (j))];
        sigma_out[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + ((gid % 4)))] += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (k) + (j)) + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (j)) + ((gid % 4)))] -= sigma_in[(((gid / 4) * 4 * 2 * dim * dim) + 2 * 4 * (dim * (i) + (k)) + 4 + ((gid % 4)))] * hamiltonian[((k) * dim + (j))];
      }
    }
  }
}
__global__
void comm_aosoa_naive_constants_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 7; ++k) {
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))] -= hamiltonian[((i) * 7 + (k))] * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))] += sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4)))] * hamiltonian[((k) * 7 + (j))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))] += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))] -= sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4)))] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))] += hamiltonian[((i) * 7 + (k))] * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))] -= sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4)))] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))] += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))] -= sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4)))] * hamiltonian[((k) * 7 + (j))];
      }
    }
  }
}
__global__
void comm_aosoa_naive_constants_direct_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 7; ++i) {
    for (int k = 0; k < 7; ++k) {
      real_t ham_real_tmp = hamiltonian[((i) * 7 + (k))];
      real_t ham_imag_tmp = hamiltonian[(7 * 7 + (i) * 7 + (k))];
      real_t sigma_real_tmp = sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + ((gid % 4)))];
      real_t sigma_imag_tmp = sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (k)) + 4 + ((gid % 4)))];
      for (int j = 0; j < 7; ++j) {
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))] -= ham_real_tmp * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))] += sigma_real_tmp * hamiltonian[((k) * 7 + (j))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))] += ham_imag_tmp * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + 4 + ((gid % 4)))] -= sigma_imag_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))] += ham_real_tmp * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + 4 + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))] -= sigma_real_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))] += ham_imag_tmp * sigma_in[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (k) + (j)) + ((gid % 4)))];
        sigma_out[(((gid / 4) * 4 * 2 * 7 * 7) + 2 * 4 * (7 * (i) + (j)) + ((gid % 4)))] -= sigma_imag_tmp * hamiltonian[((k) * 7 + (j))];
      }
    }
  }
}
__global__
void comm_aosoa (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      real_t tmp_real = 0.0;
      real_t tmp_imag = 0.0;
      for (int k = 0; k < dim; ++k) {
        tmp_imag -= hamiltonian[((i) * dim + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + threadIdx.x)];
        tmp_imag += sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + threadIdx.x)] * hamiltonian[((k) * dim + (j))];
        tmp_imag += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + 4 + threadIdx.x)];
        tmp_imag -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + 4 + threadIdx.x)] * hamiltonian[(dim * dim + (k) * dim + (j))];
        tmp_real += hamiltonian[((i) * dim + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + 4 + threadIdx.x)];
        tmp_real -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + threadIdx.x)] * hamiltonian[(dim * dim + (k) * dim + (j))];
        tmp_real += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + threadIdx.x)];
        tmp_real -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + 4 + threadIdx.x)] * hamiltonian[((k) * dim + (j))];
      }
      sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + threadIdx.x)] += tmp_real;
      sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + 4 + threadIdx.x)] += tmp_imag;
    }
  }
}
__global__
void comm_aosoa_constants (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      real_t tmp_real = 0.0;
      real_t tmp_imag = 0.0;
      for (int k = 0; k < 7; ++k) {
        tmp_imag -= hamiltonian[((i) * 7 + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + threadIdx.x)];
        tmp_imag += sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + threadIdx.x)] * hamiltonian[((k) * 7 + (j))];
        tmp_imag += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + threadIdx.x)];
        tmp_imag -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + threadIdx.x)] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += hamiltonian[((i) * 7 + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + threadIdx.x)];
        tmp_real -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + threadIdx.x)] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + threadIdx.x)];
        tmp_real -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + threadIdx.x)] * hamiltonian[((k) * 7 + (j))];
      }
      sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)] += tmp_real;
      sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)] += tmp_imag;
    }
  }
}
__global__
void comm_aosoa_constants_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  for (int i = 0; i < 7; ++i) {
    for (int k = 0; k < 7; ++k) {
      real_t ham_real_tmp = hamiltonian[((i) * 7 + (k))];
      real_t ham_imag_tmp = hamiltonian[(7 * 7 + (i) * 7 + (k))];
      real_t sigma_real_tmp = sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + threadIdx.x)];
      real_t sigma_imag_tmp = sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + threadIdx.x)];
      for (int j = 0; j < 7; ++j) {
        real_t tmp_real = sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)];
        real_t tmp_imag = sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)];
        tmp_imag -= ham_real_tmp * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + threadIdx.x)];
        tmp_imag += sigma_real_tmp * hamiltonian[((k) * 7 + (j))];
        tmp_imag += ham_imag_tmp * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + threadIdx.x)];
        tmp_imag -= sigma_imag_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += ham_real_tmp * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + threadIdx.x)];
        tmp_real -= sigma_real_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += ham_imag_tmp * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + threadIdx.x)];
        tmp_real -= sigma_imag_tmp * hamiltonian[((k) * 7 + (j))];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)] = tmp_real;
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)] = tmp_imag;
      }
    }
  }
}
__global__
void comm_aosoa_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + 4 + threadIdx.x)] -= hamiltonian[((i) * dim + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + 4 + threadIdx.x)] += sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + threadIdx.x)] * hamiltonian[((k) * dim + (j))];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + 4 + threadIdx.x)] += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + 4 + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + 4 + threadIdx.x)] -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + 4 + threadIdx.x)] * hamiltonian[(dim * dim + (k) * dim + (j))];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + threadIdx.x)] += hamiltonian[((i) * dim + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + 4 + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + threadIdx.x)] -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + threadIdx.x)] * hamiltonian[(dim * dim + (k) * dim + (j))];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + threadIdx.x)] += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (k) + (j)) + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (j)) + threadIdx.x)] -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * dim * dim)) + 2 * 4 * (dim * (i) + (k)) + 4 + threadIdx.x)] * hamiltonian[((k) * dim + (j))];
      }
    }
  }
}
__global__
void comm_aosoa_constants_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 7; ++k) {
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)] -= hamiltonian[((i) * 7 + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)] += sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + threadIdx.x)] * hamiltonian[((k) * 7 + (j))];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)] += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)] -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + threadIdx.x)] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)] += hamiltonian[((i) * 7 + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)] -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + threadIdx.x)] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)] += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)] -= sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + threadIdx.x)] * hamiltonian[((k) * 7 + (j))];
      }
    }
  }
}
__global__
void comm_aosoa_constants_direct_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  for (int i = 0; i < 7; ++i) {
    for (int k = 0; k < 7; ++k) {
      real_t ham_real_tmp = hamiltonian[((i) * 7 + (k))];
      real_t ham_imag_tmp = hamiltonian[(7 * 7 + (i) * 7 + (k))];
      real_t sigma_real_tmp = sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + threadIdx.x)];
      real_t sigma_imag_tmp = sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (k)) + 4 + threadIdx.x)];
      for (int j = 0; j < 7; ++j) {
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)] -= ham_real_tmp * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)] += sigma_real_tmp * hamiltonian[((k) * 7 + (j))];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)] += ham_imag_tmp * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + 4 + threadIdx.x)] -= sigma_imag_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)] += ham_real_tmp * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + 4 + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)] -= sigma_real_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)] += ham_imag_tmp * sigma_in[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (k) + (j)) + threadIdx.x)];
        sigma_out[(((64 * blockIdx.y + threadIdx.y) * (4 * 2 * 7 * 7)) + 2 * 4 * (7 * (i) + (j)) + threadIdx.x)] -= sigma_imag_tmp * hamiltonian[((k) * 7 + (j))];
      }
    }
  }
}
__global__
void comm_manual_aosoa (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      real_vec_t tmp_real = v(0.0);
      real_vec_t tmp_imag = v(0.0);
      for (int k = 0; k < dim; ++k) {
        tmp_imag -= hamiltonian[((i) * dim + (k))] * sigma_in[((gid * dim * dim * 2) + 2 * (dim * (k) + (j)))];
        tmp_imag += sigma_in[((gid * dim * dim * 2) + 2 * (dim * (i) + (k)))] * hamiltonian[((k) * dim + (j))];
        tmp_imag += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[((gid * dim * dim * 2) + 2 * (dim * (k) + (j)) + 1)];
        tmp_imag -= sigma_in[((gid * dim * dim * 2) + 2 * (dim * (i) + (k)) + 1)] * hamiltonian[(dim * dim + (k) * dim + (j))];
        tmp_real += hamiltonian[((i) * dim + (k))] * sigma_in[((gid * dim * dim * 2) + 2 * (dim * (k) + (j)) + 1)];
        tmp_real -= sigma_in[((gid * dim * dim * 2) + 2 * (dim * (i) + (k)))] * hamiltonian[(dim * dim + (k) * dim + (j))];
        tmp_real += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[((gid * dim * dim * 2) + 2 * (dim * (k) + (j)))];
        tmp_real -= sigma_in[((gid * dim * dim * 2) + 2 * (dim * (i) + (k)) + 1)] * hamiltonian[((k) * dim + (j))];
      }
      sigma_out[((gid * dim * dim * 2) + 2 * (dim * (i) + (j)))] += tmp_real;
      sigma_out[((gid * dim * dim * 2) + 2 * (dim * (i) + (j)) + 1)] += tmp_imag;
    }
  }
}
__global__
void comm_manual_aosoa_constants (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      real_vec_t tmp_real = v(0.0);
      real_vec_t tmp_imag = v(0.0);
      for (int k = 0; k < 7; ++k) {
        tmp_imag -= hamiltonian[((i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        tmp_imag += sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)))] * hamiltonian[((k) * 7 + (j))];
        tmp_imag += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        tmp_imag -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1)] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += hamiltonian[((i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        tmp_real -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)))] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        tmp_real -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1)] * hamiltonian[((k) * 7 + (j))];
      }
      sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] += tmp_real;
      sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] += tmp_imag;
    }
  }
}
__global__
void comm_manual_aosoa_constants_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 7; ++i) {
    for (int k = 0; k < 7; ++k) {
      real_vec_t ham_real_tmp = v(hamiltonian[((i) * 7 + (k))]);
      real_vec_t ham_imag_tmp = v(hamiltonian[(7 * 7 + (i) * 7 + (k))]);
      real_vec_t sigma_real_tmp = sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)))];
      real_vec_t sigma_imag_tmp = sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1)];
      for (int j = 0; j < 7; ++j) {
        real_vec_t tmp_real = sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))];
        real_vec_t tmp_imag = sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)];
        tmp_imag -= ham_real_tmp * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        tmp_imag += sigma_real_tmp * hamiltonian[((k) * 7 + (j))];
        tmp_imag += ham_imag_tmp * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        tmp_imag -= sigma_imag_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += ham_real_tmp * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        tmp_real -= sigma_real_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += ham_imag_tmp * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        tmp_real -= sigma_imag_tmp * hamiltonian[((k) * 7 + (j))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] = tmp_real;
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] = tmp_imag;
      }
    }
  }
}
__global__
void comm_manual_aosoa_constants_perm_prefetch (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 7; ++i) {
    int j = 0;
    for (j = 0; j < 7; ++j) {
      real_vec_t tmp_real = v(0.0);
      real_vec_t tmp_imag = v(0.0);
      for (int k = 0; k < 7; ++k) {
        tmp_imag -= hamiltonian[((i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        tmp_imag += sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)))] * hamiltonian[((k) * 7 + (j))];
        tmp_imag += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        tmp_imag -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1)] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += hamiltonian[((i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        tmp_real -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)))] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        tmp_real += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        tmp_real -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1)] * hamiltonian[((k) * 7 + (j))];
      }
      sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] += tmp_real;
      sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] += tmp_imag;
    }
  }
}
__global__
void comm_manual_aosoa_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        sigma_out[((gid * dim * dim * 2) + 2 * (dim * (i) + (j)) + 1)] -= hamiltonian[((i) * dim + (k))] * sigma_in[((gid * dim * dim * 2) + 2 * (dim * (k) + (j)))];
        sigma_out[((gid * dim * dim * 2) + 2 * (dim * (i) + (j)) + 1)] += sigma_in[((gid * dim * dim * 2) + 2 * (dim * (i) + (k)))] * hamiltonian[((k) * dim + (j))];
        sigma_out[((gid * dim * dim * 2) + 2 * (dim * (i) + (j)) + 1)] += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[((gid * dim * dim * 2) + 2 * (dim * (k) + (j)) + 1)];
        sigma_out[((gid * dim * dim * 2) + 2 * (dim * (i) + (j)) + 1)] -= sigma_in[((gid * dim * dim * 2) + 2 * (dim * (i) + (k)) + 1)] * hamiltonian[(dim * dim + (k) * dim + (j))];
        sigma_out[((gid * dim * dim * 2) + 2 * (dim * (i) + (j)))] += hamiltonian[((i) * dim + (k))] * sigma_in[((gid * dim * dim * 2) + 2 * (dim * (k) + (j)) + 1)];
        sigma_out[((gid * dim * dim * 2) + 2 * (dim * (i) + (j)))] -= sigma_in[((gid * dim * dim * 2) + 2 * (dim * (i) + (k)))] * hamiltonian[(dim * dim + (k) * dim + (j))];
        sigma_out[((gid * dim * dim * 2) + 2 * (dim * (i) + (j)))] += hamiltonian[(dim * dim + (i) * dim + (k))] * sigma_in[((gid * dim * dim * 2) + 2 * (dim * (k) + (j)))];
        sigma_out[((gid * dim * dim * 2) + 2 * (dim * (i) + (j)))] -= sigma_in[((gid * dim * dim * 2) + 2 * (dim * (i) + (k)) + 1)] * hamiltonian[((k) * dim + (j))];
      }
    }
  }
}
__global__
void comm_manual_aosoa_constants_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 7; ++k) {
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] -= hamiltonian[((i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] += sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)))] * hamiltonian[((k) * 7 + (j))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1)] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] += hamiltonian[((i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)))] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1)] * hamiltonian[((k) * 7 + (j))];
      }
    }
  }
}
__global__
void comm_manual_aosoa_constants_direct_prefetch (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 7; ++i) {
    int j = 0;
    for (j = 0; j < 7; ++j) {
      for (int k = 0; k < 7; ++k)
      {
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] -= hamiltonian[((i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] += sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)))] * hamiltonian[((k) * 7 + (j))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1)] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] += hamiltonian[((i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)))] * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] += hamiltonian[(7 * 7 + (i) * 7 + (k))] * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] -= sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1)] * hamiltonian[((k) * 7 + (j))];
      }
    }
  }
}
__global__
void comm_manual_aosoa_constants_direct_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out,
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 7; ++i) {
    for (int k = 0; k < 7; ++k) {
      real_vec_t ham_real_tmp = v(hamiltonian[((i) * 7 + (k))]);
      real_vec_t ham_imag_tmp = v(hamiltonian[(7 * 7 + (i) * 7 + (k))]);
      real_vec_t sigma_real_tmp = sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)))];
      real_vec_t sigma_imag_tmp = sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (k)) + 1)];
      for (int j = 0; j < 7; ++j) {
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] -= ham_real_tmp * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] += sigma_real_tmp * hamiltonian[((k) * 7 + (j))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] += ham_imag_tmp * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)) + 1)] -= sigma_imag_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] += ham_real_tmp * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)) + 1)];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] -= sigma_real_tmp * hamiltonian[(7 * 7 + (k) * 7 + (j))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] += ham_imag_tmp * sigma_in[((gid * 7 * 7 * 2) + 2 * (7 * (k) + (j)))];
        sigma_out[((gid * 7 * 7 * 2) + 2 * (7 * (i) + (j)))] -= sigma_imag_tmp * hamiltonian[((k) * 7 + (j))];
      }
    }
  }
}
__global__
void final_gpu_kernel (
    const real_2_t *__restrict__ sigma_in,
          real_2_t *__restrict__ sigma_out,
    const real_2_t *__restrict__ hamiltonian,
    const int num)
{
  __shared__ real_t ham_local_real[7*7];
  __shared__ real_t ham_local_imag[7*7];
  __shared__ real_t sigma_local_real[2][2][7*7];
  __shared__ real_t sigma_local_imag[2][2][7*7];
  int ij = threadIdx.x;
  int i = ij / 7;
  int j = ij % 7;
  int sub_group_id = threadIdx.y;
  int start = blockIdx.x * 2 * 16 + sub_group_id * 16;
  int stop = ((num) < (start + 16) ? (num) : (start + 16));
  real_2_t snew1_ij, snew2_ij;
  real_2_t s1, s2;
  if (ij < (7 * 7) && sub_group_id == 0)
  {
    const real_2_t h = hamiltonian[ij];
    ham_local_real[ij] = h.x;
    ham_local_imag[ij] = h.y;
  }
  for (int m = start; m < stop; m += 2)
  {
    __syncthreads();
    if (ij < (7 * 7))
    {
      s1 = sigma_in[((m) * 7 * 7 + ((i) * 7 + (j)))];
      sigma_local_real[0][sub_group_id][ij] = s1.x;
      sigma_local_imag[0][sub_group_id][ij] = s1.y;
      s2 = sigma_in[((m + 1) * 7 * 7 + ((i) * 7 + (j)))];
      sigma_local_real[1][sub_group_id][ij] = s2.x;
      sigma_local_imag[1][sub_group_id][ij] = s2.y;
      s1 = sigma_out[((m) * 7 * 7 + ((i) * 7 + (j)))];
      snew1_ij.x = s1.x;
      snew2_ij.x = s1.y;
      s2 = sigma_out[((m + 1) * 7 * 7 + ((i) * 7 + (j)))];
      snew1_ij.y = s2.x;
      snew2_ij.y = s2.y;
    }
    __syncthreads();
    if (ij < (7 * 7))
    {
      for (int k = 0; k < 7; ++k)
      {
        const int ik = ((i) * 7 + (k));
        const int kj = ((k) * 7 + (j));
        s1 = {sigma_local_real[0][sub_group_id][kj], sigma_local_real[1][sub_group_id][kj]};
        s2 = {sigma_local_imag[0][sub_group_id][kj], sigma_local_imag[1][sub_group_id][kj]};
        snew1_ij += ham_local_real[ik] * s2;
        snew1_ij += ham_local_imag[ik] * s1;
        snew2_ij -= ham_local_real[ik] * s1;
        snew2_ij += ham_local_imag[ik] * s2;
        s1 = {sigma_local_real[0][sub_group_id][ik], sigma_local_real[1][sub_group_id][ik]};
        s2 = {sigma_local_imag[0][sub_group_id][ik], sigma_local_imag[1][sub_group_id][ik]};
        snew1_ij -= ham_local_real[kj] * s2;
        snew1_ij += ham_local_imag[kj] * s1;
        snew2_ij += ham_local_real[kj] * s1;
        snew2_ij -= ham_local_imag[kj] * s2;
      }
      sigma_out[((m) * 7 * 7 + ((i) * 7 + (j)))] = {snew1_ij.x, snew2_ij.x};
      sigma_out[((m + 1) * 7 * 7 + ((i) * 7 + (j)))] = {snew1_ij.y, snew2_ij.y};
    }
  }
}
