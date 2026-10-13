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

#include "vector.h"
#include <string.h>    // memcpy
#define BLOCK_SIZE 256



void Vector::initialize_gpu(int n)
{
  this->n = n;
  array_size = n * sizeof(lsqt_real);
  real_part=std::make_shared<LBuffer>(array_size);
  imag_part=std::make_shared<LBuffer>(array_size);
}

Vector::Vector(int n)
{
  initialize_gpu(n);
  {
    size_t gws[]={size_t((n - 1) / BLOCK_SIZE + 1)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &real_part->mem, &imag_part->mem};
    int info[]={sizeof(n), iris_rw, iris_rw};
    iris::Task task;
    iris_check(task.kernel("gpu_set_zero",1,nullptr,gws,lws,3,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}



Vector::Vector(Vector& original)
{
  // Just teach myself: one can access private members of another instance
  // of the class from within the class
  initialize_gpu(original.n);
  {
    size_t gws[]={size_t((n - 1) / BLOCK_SIZE + 1)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &original.real_part->mem, &original.imag_part->mem, &real_part->mem, &imag_part->mem};
    int info[]={sizeof(n), iris_r, iris_r, iris_rw, iris_rw};
    iris::Task task;
    iris_check(task.kernel("gpu_copy_state",1,nullptr,gws,lws,5,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}

Vector::~Vector()
{ real_part.reset(); imag_part.reset();
}



void Vector::add(Vector& other)
{
  {
    size_t gws[]={size_t((n - 1) / BLOCK_SIZE + 1)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &other.real_part->mem, &other.imag_part->mem, &real_part->mem, &imag_part->mem};
    int info[]={sizeof(n), iris_r, iris_r, iris_rw, iris_rw};
    iris::Task task;
    iris_check(task.kernel("gpu_add_state",1,nullptr,gws,lws,5,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}

void Vector::copy(Vector& other)
{
  {
    size_t gws[]={size_t((n - 1) / BLOCK_SIZE + 1)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &other.real_part->mem, &other.imag_part->mem, &real_part->mem, &imag_part->mem};
    int info[]={sizeof(n), iris_r, iris_r, iris_rw, iris_rw};
    iris::Task task;
    iris_check(task.kernel("gpu_copy_state",1,nullptr,gws,lws,5,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}



void Vector::apply_sz(Vector& other)
{
  {
    size_t gws[]={size_t((n - 1) / BLOCK_SIZE + 1)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&n, &other.real_part->mem, &other.imag_part->mem, &real_part->mem, &imag_part->mem};
    int info[]={sizeof(n), iris_r, iris_r, iris_rw, iris_rw};
    iris::Task task;
    iris_check(task.kernel("gpu_apply_sz",1,nullptr,gws,lws,5,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}

void Vector::copy_from_host(lsqt_real* other_real, lsqt_real* other_imag)
{
  real_part->update(other_real);
  imag_part->update(other_imag);
}

void Vector::copy_to_host(lsqt_real* target_real, lsqt_real* target_imag)
{
  real_part->copy_to(target_real);
  imag_part->copy_to(target_imag);
}

void Vector::swap(Vector& other)
{
  auto tmp_real = real_part;
  auto tmp_imag = imag_part;
  real_part = other.real_part, imag_part = other.imag_part;
  other.real_part = tmp_real;
  other.imag_part = tmp_imag;
}





void Vector::inner_product_1(int number_of_atoms, Vector& other, Vector& target, int offset)
{
  int grid_size = (number_of_atoms - 1) / BLOCK_SIZE + 1;
  {
    size_t gws[]={size_t(grid_size)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&number_of_atoms, &real_part->mem, &imag_part->mem, &other.real_part->mem, &other.imag_part->mem, &target.real_part->mem, &target.imag_part->mem, &offset};
    int info[]={sizeof(number_of_atoms), iris_r, iris_r, iris_r, iris_r, iris_rw, iris_rw, sizeof(offset)};
    iris::Task task;
    iris_check(task.kernel("gpu_find_inner_product_1",1,nullptr,gws,lws,8,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}



void Vector::inner_product_2(int number_of_atoms, int number_of_moments, Vector& target)
{
  {
    size_t gws[]={size_t(number_of_moments)*BLOCK_SIZE},lws[]={BLOCK_SIZE};
    void *args[]={&number_of_atoms, &real_part->mem, &imag_part->mem, &target.real_part->mem, &target.imag_part->mem};
    int info[]={sizeof(number_of_atoms), iris_r, iris_r, iris_rw, iris_rw};
    iris::Task task;
    iris_check(task.kernel("gpu_find_inner_product_2",1,nullptr,gws,lws,5,args,info));
    iris_check(task.submit(iris_gpu,nullptr,true));
  }
}
