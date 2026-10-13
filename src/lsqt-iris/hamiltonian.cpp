/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, and Ari Harju

    This file is part of GPUQT.

    GPUQT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUQT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUQT.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "hamiltonian.h"
#include "model.h"
#include "vector.h"
#include <string.h>    // memcpy
#define BLOCK_SIZE 256

void Hamiltonian::initialize_gpu(Model& model)
{
  n = model.number_of_atoms;
  max_neighbor = model.max_neighbor;
  energy_max = model.energy_max;
  grid_size = (model.number_of_atoms - 1) / BLOCK_SIZE + 1;

  neighbor_number=std::make_shared<LBuffer>(sizeof(int) * n);
  neighbor_list=std::make_shared<LBuffer>(sizeof(int) * model.number_of_pairs);
  potential=std::make_shared<LBuffer>(sizeof(lsqt_real) * n);
  hopping_real=std::make_shared<LBuffer>(sizeof(lsqt_real) * model.number_of_pairs);
  hopping_imag=std::make_shared<LBuffer>(sizeof(lsqt_real) * model.number_of_pairs);
  xx=std::make_shared<LBuffer>(sizeof(lsqt_real) * model.number_of_pairs);

  neighbor_number->update(model.neighbor_number);
  delete[] model.neighbor_number;
  potential->update(model.potential);
  delete[] model.potential;

  int* neighbor_list_new = new int[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      neighbor_list_new[m * n + i] = model.neighbor_list[i * max_neighbor + m];
    }
  }
  delete[] model.neighbor_list;
  neighbor_list->update(neighbor_list_new);
  delete[] neighbor_list_new;

  lsqt_real* hopping_real_new = new lsqt_real[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      hopping_real_new[m * n + i] = model.hopping_real[i * max_neighbor + m];
    }
  }
  delete[] model.hopping_real;
  hopping_real->update(hopping_real_new);
  delete[] hopping_real_new;

  lsqt_real* hopping_imag_new = new lsqt_real[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      hopping_imag_new[m * n + i] = model.hopping_imag[i * max_neighbor + m];
    }
  }
  delete[] model.hopping_imag;
  hopping_imag->update(hopping_imag_new);
  delete[] hopping_imag_new;

  lsqt_real* xx_new = new lsqt_real[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      xx_new[m * n + i] = model.xx[i * max_neighbor + m];
    }
  }
  delete[] model.xx;
  xx->update(xx_new);
  delete[] xx_new;
}

Hamiltonian::Hamiltonian(Model& model)
{
  initialize_gpu(model);
}

Hamiltonian::~Hamiltonian()
{ neighbor_number.reset(); neighbor_list.reset(); potential.reset(); hopping_real.reset(); hopping_imag.reset(); xx.reset();
}



// |output> = H |input>
void Hamiltonian::apply(Vector& input, Vector& output)
{
  {
    size_t gws[]={size_t(grid_size)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &energy_max, &neighbor_number->mem, &neighbor_list->mem, &potential->mem, &hopping_real->mem, &hopping_imag->mem, &input.real_part->mem, &input.imag_part->mem, &output.real_part->mem, &output.imag_part->mem};
    int info[]={sizeof(n), sizeof(energy_max), iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_rw, iris_rw};
    iris::Task task;
    iris_check(task.kernel("gpu_apply_hamiltonian",1,nullptr,gws,lws,11,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}



// |output> = [X, H] |input>
void Hamiltonian::apply_commutator(Vector& input, Vector& output)
{
  {
    size_t gws[]={size_t(grid_size)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &energy_max, &neighbor_number->mem, &neighbor_list->mem, &hopping_real->mem, &hopping_imag->mem, &xx->mem, &input.real_part->mem, &input.imag_part->mem, &output.real_part->mem, &output.imag_part->mem};
    int info[]={sizeof(n), sizeof(energy_max), iris_rw, iris_rw, iris_rw, iris_rw, iris_rw, iris_rw, iris_rw, iris_rw, iris_rw};
    iris::Task task;
    iris_check(task.kernel("gpu_apply_commutator",1,nullptr,gws,lws,11,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}



// |output> = V |input>
void Hamiltonian::apply_current(Vector& input, Vector& output)
{
  {
    size_t gws[]={size_t(grid_size)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &neighbor_number->mem, &neighbor_list->mem, &hopping_real->mem, &hopping_imag->mem, &xx->mem, &input.real_part->mem, &input.imag_part->mem, &output.real_part->mem, &output.imag_part->mem};
    int info[]={sizeof(n), iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_rw, iris_rw};
    iris::Task task;
    iris_check(task.kernel("gpu_apply_current",1,nullptr,gws,lws,10,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}

// Kernel which calculates the two first terms of time evolution as described by
// Eq. (36) in [Comput. Phys. Commun.185, 28 (2014)].


// Wrapper for the kernel above
void Hamiltonian::chebyshev_01(
  Vector& state_0, Vector& state_1, Vector& state, lsqt_real bessel_0, lsqt_real bessel_1, int direction)
{
  {
    size_t gws[]={size_t(grid_size)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &state_0.real_part->mem, &state_0.imag_part->mem, &state_1.real_part->mem, &state_1.imag_part->mem, &state.real_part->mem, &state.imag_part->mem, &bessel_0, &bessel_1, &direction};
    int info[]={sizeof(n), iris_r, iris_r, iris_r, iris_r, iris_rw, iris_rw, sizeof(bessel_0), sizeof(bessel_1), sizeof(direction)};
    iris::Task task;
    iris_check(task.kernel("gpu_chebyshev_01",1,nullptr,gws,lws,10,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}

// Kernel for calculating further terms of Eq. (36)
// in [Comput. Phys. Commun.185, 28 (2014)].


// Wrapper for the kernel above
void Hamiltonian::chebyshev_2(
  Vector& state_0, Vector& state_1, Vector& state_2, Vector& state, lsqt_real bessel_m, int label)
{
  {
    size_t gws[]={size_t(grid_size)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &energy_max, &neighbor_number->mem, &neighbor_list->mem, &potential->mem, &hopping_real->mem, &hopping_imag->mem, &state_0.real_part->mem, &state_0.imag_part->mem, &state_1.real_part->mem, &state_1.imag_part->mem, &state_2.real_part->mem, &state_2.imag_part->mem, &state.real_part->mem, &state.imag_part->mem, &bessel_m, &label};
    int info[]={sizeof(n), sizeof(energy_max), iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_rw, iris_rw, iris_rw, iris_rw, sizeof(bessel_m), sizeof(label)};
    iris::Task task;
    iris_check(task.kernel("gpu_chebyshev_2",1,nullptr,gws,lws,17,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}

// Kernel which calculates the two first terms of commutator [X, U(dt)]
// Corresponds to Eq. (37) in [Comput. Phys. Commun.185, 28 (2014)].


// Wrapper for kernel above
void Hamiltonian::chebyshev_1x(Vector& input, Vector& output, lsqt_real bessel_1)
{
  {
    size_t gws[]={size_t(grid_size)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &input.real_part->mem, &input.imag_part->mem, &output.real_part->mem, &output.imag_part->mem, &bessel_1};
    int info[]={sizeof(n), iris_r, iris_r, iris_rw, iris_rw, sizeof(bessel_1)};
    iris::Task task;
    iris_check(task.kernel("gpu_chebyshev_1x",1,nullptr,gws,lws,6,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}

// Kernel which calculates the further terms of [X, U(dt)]


// Wrapper for the kernel above
void Hamiltonian::chebyshev_2x(
  Vector& state_0,
  Vector& state_0x,
  Vector& state_1,
  Vector& state_1x,
  Vector& state_2,
  Vector& state_2x,
  Vector& state,
  lsqt_real bessel_m,
  int label)
{
  {
    size_t gws[]={size_t(grid_size)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &energy_max, &neighbor_number->mem, &neighbor_list->mem, &potential->mem, &hopping_real->mem, &hopping_imag->mem, &xx->mem, &state_0.real_part->mem, &state_0.imag_part->mem, &state_0x.real_part->mem, &state_0x.imag_part->mem, &state_1.real_part->mem, &state_1.imag_part->mem, &state_1x.real_part->mem, &state_1x.imag_part->mem, &state_2.real_part->mem, &state_2.imag_part->mem, &state_2x.real_part->mem, &state_2x.imag_part->mem, &state.real_part->mem, &state.imag_part->mem, &bessel_m, &label};
    int info[]={sizeof(n), sizeof(energy_max), iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_rw, iris_rw, iris_rw, iris_rw, iris_rw, iris_rw, sizeof(bessel_m), sizeof(label)};
    iris::Task task;
    iris_check(task.kernel("gpu_chebyshev_2x",1,nullptr,gws,lws,24,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}

// Kernel for doing the Chebyshev iteration phi_2 = 2 * H * phi_1 - phi_0.


// Wrapper for the Chebyshev iteration
void Hamiltonian::kernel_polynomial(Vector& state_0, Vector& state_1, Vector& state_2)
{
  {
    size_t gws[]={size_t(grid_size)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &energy_max, &neighbor_number->mem, &neighbor_list->mem, &potential->mem, &hopping_real->mem, &hopping_imag->mem, &state_0.real_part->mem, &state_0.imag_part->mem, &state_1.real_part->mem, &state_1.imag_part->mem, &state_2.real_part->mem, &state_2.imag_part->mem};
    int info[]={sizeof(n), sizeof(energy_max), iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_r, iris_rw, iris_rw};
    iris::Task task;
    iris_check(task.kernel("gpu_kernel_polynomial",1,nullptr,gws,lws,13,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}
