/* -------------------------------------------------------------------------- */
/* Rainflow cycle counting algorithm according to:                            */
/* ASTM E1049-85,                                                             */
/* Standard Practices for Cycle Counting in Fatigue Analysis,                 */
/* ASTM International. (https://doi.org/10.1520/E1049-85R17)                  */
/*                                                                            */
/* By: Carlos Souto - csouto@fe.up.pt                                         */
/* -------------------------------------------------------------------------- */
/* Extracted from ASTM E1049-85 – Rainflow Counting:                          */
/* Rules for this method are as follows: let X denote range under             */
/* consideration; Y, previous range adjacent to X; and S, starting point in   */
/* the history.                                                               */
/* (1) Read next peak or valley. If out of data, go to Step 6.                */
/* (2) If there are less than three points, go to Step 1. Form ranges X and Y */
/*     using the three most recent peaks and valleys that have not been       */
/*     discarded.                                                             */
/* (3) Compare the absolute values of ranges X and Y.                         */
/*     (a) If X < Y, go to Step 1.                                            */
/*     (b) If X >= Y, go to Step 4.                                           */
/* (4) If range Y contains the starting point S, go to Step 5; otherwise,     */
/*     count range Y as one cycle; discard the peak and valley of Y; and go   */
/*     to Step 2.                                                             */
/* (5) Count range Y as one-half cycle; discard the first point (peak or      */
/*     valley) in range Y; move the starting point to the second point in     */
/*     range Y; and go to Step 2.                                             */
/* (6) Count each range that has not been previously counted as one-half      */
/*     cycle.                                                                 */
/* -------------------------------------------------------------------------- */
/* License: BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause       */
/*                                                                            */
/* Copyright (c) 2021, Carlos Daniel Santos Souto.                            */
/* All rights reserved.                                                       */
/*                                                                            */
/* Redistribution and use in source and binary forms, with or without         */
/* modification, are permitted provided that the following conditions are     */
/* met:                                                                       */
/*                                                                            */
/* 1. Redistributions of source code must retain the above copyright notice,  */
/*    this list of conditions and the following disclaimer.                   */
/* 2. Redistributions in binary form must reproduce the above copyright       */
/*    notice, this list of conditions and the following disclaimer in the     */
/*    documentation and/or other materials provided with the distribution.    */
/* 3. Neither the name of the copyright holder nor the names of its           */
/*    contributors may be used to endorse or promote products derived from    */
/*    this software without specific prior written permission.                */
/*                                                                            */
/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS    */
/* IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,  */
/* THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR     */
/* PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR          */
/* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,      */
/* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,        */
/* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR         */
/* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF     */
/* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       */
/* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         */
/* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               */
/* -------------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "iris_dmem.hpp"
#include <chrono>
#include "types.hpp"
#include "reference.h"

/// <summary>
/// Utility function to get the local extrema (maxima and minima) of the provided 
/// stress-time history as a sequence of peaks and valleys (bad in-between data 
/// points – that are neither peaks nor valleys – are removed).
/// </summary>
/// <param name="history">The provided stress-time history.</param>
/// <returns>The obtained sequence of peaks and valleys (local extrema).</returns>
int main(int argc, char* argv[]) {
  IRISSession session(&argc,&argv);
  const int num_history = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int *history_lengths = (int*) malloc ((num_history + 1) * sizeof(int));
  int *result_lengths = (int*) malloc (num_history * sizeof(int));
  int *ref_result_lengths = (int*) malloc (num_history * sizeof(int));
  
  srand(123);

  // initialize history length with a multiple of the scale unit
  const int scale = 100;
  size_t total_length  = 0;

  int n;
  for (n = 0; n < num_history; n++) {
     history_lengths[n] = total_length;
     total_length += (rand() % 10 + 1) * scale;
  }
  history_lengths[n] = total_length;
  
  printf("Total history length = %zu\n", total_length);

  double *history = (double*) malloc (total_length * sizeof(double));
  for (size_t i = 0; i < total_length; i++) {
    history[i] = rand() / (double)RAND_MAX;
  }

  double *extrema = (double*) malloc (total_length * sizeof(double));
  RFResult *results = (RFResult*) malloc (total_length * sizeof(RFResult));
  int *points = (int*) malloc (total_length * sizeof(int));
  
  int *d_history_lengths;
  iris_alloc ((void**)&d_history_lengths, (num_history + 1) * sizeof(int));
  iris_copy(d_history_lengths, history_lengths, (num_history  + 1) * sizeof(int), IRISHostToDevice);

  int *d_result_lengths;
  iris_alloc((void**)&d_result_lengths, num_history * sizeof(int));

  double *d_history;
  iris_alloc((void**)&d_history, total_length * sizeof(double));
  iris_copy(d_history, history, total_length * sizeof(double), IRISHostToDevice);

  double *d_extrema;
  iris_alloc((void**)&d_extrema, total_length * sizeof(double));

  RFResult *d_results;
  iris_alloc((void**)&d_results, total_length * sizeof(RFResult));

  int *d_points;
  iris_alloc((void**)&d_points, total_length * sizeof(int));
  
  dim3 blocks (256);
  dim3 grids (num_history / 256 + 1);

  iris_sync();
  auto start = std::chrono::steady_clock::now();

  for (n = 0; n < repeat; n++) {
    iris_launch("rainflow_count", grids, blocks, 
      d_history, 
      d_history_lengths,
      d_extrema,
      d_points,
      d_results,
      d_result_lengths,
      num_history
    );
  }

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  iris_copy(result_lengths, d_result_lengths, num_history * sizeof(int), IRISDeviceToHost);

  reference (
    history, 
    history_lengths,
    extrema,
    points,
    results,
    ref_result_lengths,
    num_history
  );

  int error = memcmp(ref_result_lengths, result_lengths, num_history * sizeof(int));
  printf("%s\n", error ? "FAIL" : "PASS");

  iris_free(d_history);
  iris_free(d_history_lengths);
  iris_free(d_extrema);
  iris_free(d_points);
  iris_free(d_results);
  iris_free(d_result_lengths);
  free(history);
  free(history_lengths);
  free(extrema);
  free(points);
  free(results);
  free(result_lengths);
  free(ref_result_lengths);

  return 0;
}
