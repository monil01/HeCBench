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
#include <random>

class Charge
{
public:
  void add_impurities(
    std::mt19937&,
    int,
    lsqt_real*,
    int*,
    std::vector<lsqt_real>&,
    std::vector<lsqt_real>&,
    std::vector<lsqt_real>&,
    lsqt_real*);
  bool has = false;
  int Ni;  // number of impurities
  lsqt_real W;  // impurity strength
  lsqt_real xi; // impurity range
private:
  int Nx, Ny, Nz, Nxyz; // number of cells
  lsqt_real rc;              // cutoff distance for impurity potential
  lsqt_real rc2;             // cutoff square
  std::vector<int> cell_count;
  std::vector<int> cell_count_sum;
  std::vector<int> cell_contents;
  std::vector<int> impurity_indices;
  std::vector<lsqt_real> impurity_strength;
  void find_impurity_indices(std::mt19937&, int);
  void find_impurity_strength(std::mt19937&);
  void find_potentials(
    int, lsqt_real*, int*, std::vector<lsqt_real>&, std::vector<lsqt_real>&, std::vector<lsqt_real>&, lsqt_real*);
  int find_cell_id(lsqt_real, lsqt_real, lsqt_real, lsqt_real);
  void find_cell_id(lsqt_real, lsqt_real, lsqt_real, lsqt_real, int&, int&, int&, int&);
  void find_cell_numbers(int*, lsqt_real*);
  void
  find_cell_contents(int, int*, lsqt_real*, std::vector<lsqt_real>&, std::vector<lsqt_real>&, std::vector<lsqt_real>&);
  int find_neighbor_cell(int, int, int, int, int, int, int);
};
