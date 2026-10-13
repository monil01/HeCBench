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

#pragma once
#include "common.h"

class Vector
{
public:
  Vector(int n);
  Vector(Vector& original);
  ~Vector();

  void add(Vector& other);
  void copy(Vector& other);
  void apply_sz(Vector& other);
  void copy_from_host(lsqt_real* other_real, lsqt_real* other_imag);
  void copy_to_host(lsqt_real* target_real, lsqt_real* target_imag);
  void swap(Vector& other);
  void inner_product_1(int, Vector& other, Vector& target, int offset);
  void inner_product_2(int, int, Vector& target);

  LMemory real_part;
  LMemory imag_part;

private:
  void initialize_gpu(int n);
  void initialize_cpu(int n);
  int n;
  int array_size;
};
