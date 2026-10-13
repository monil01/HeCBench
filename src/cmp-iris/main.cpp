////////////////////////////////////////////////////////////////////////////////
/**
 * @file main.cpp
 * @date 2017-03-04
 * @author Tiago Lobato Gimenes    (tlgimenes@gmail.com)
 *
 * @copyright
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
////////////////////////////////////////////////////////////////////////////////


#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iris/iris.hpp>
#include "memory.h"
#include "launch.h"

#include "log.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "su_gather.hpp"

int main(int argc, const char** argv) {
#ifdef SAVE
  std::ofstream c_out("cmp.c.su", std::ofstream::out | std::ios::binary);
  std::ofstream s_out("cmp.coher.su", std::ofstream::out | std::ios::binary);
  std::ofstream stack("cmp.stack.su", std::ofstream::out | std::ios::binary);
#endif

  iris::Platform platform; char **iris_argv=const_cast<char**>(argv); platform.init(&argc,&iris_argv,1); prove_backend();
  // Parse command line and read arguments
  parser::add_argument("-c0", "C0 constant");
  parser::add_argument("-c1", "C1 constant");
  parser::add_argument("-nc", "NC constant");
  parser::add_argument("-aph", "APH constant");
  parser::add_argument("-tau", "Tau constant");
  parser::add_argument("-i", "Data path");
  parser::add_argument("-v", "Verbosity Level 0-3");

  parser::parse(argc, argv);

  // Read parameters and input
  const real c0 = std::stof(parser::get("-c0", true)) * FACTOR;
  const real c1 = std::stof(parser::get("-c1", true)) * FACTOR;
  const real itau = std::stof(parser::get("-tau", true));
  const int nc = std::stoi(parser::get("-nc", true));
  const int aph = std::stoi(parser::get("-aph", true));
  std::string path = parser::get("-i", true);
  logger::verbosity_level(std::stoi(parser::get("-v", false)));

  // Reads *.su data and starts gather
  su_gather gather(path, aph, nc);

  real *h_gx, *h_gy, *h_sx, *h_sy, *h_scalco, *h_samples, *h_str, *h_stk, dt;
  int *ntraces_by_cdp_id, *h_ctr;

  // Linearize gather data in order to improve data coalescence in GPU
  gather.linearize(ntraces_by_cdp_id, h_samples, dt, h_gx, h_gy, h_sx, h_sy, h_scalco, nc);
  const int  ttraces = gather.ttraces(); // Total traces -> Total amount of traces read
  const int  ncdps = gather().size();    // Number of cdps -> Total number of cdps read
  const int  ns = gather.ns();           // Number of samples
  const int  ntrs = gather.ntrs();       // Max number of traces by cdp
  const real inc = (c1-c0) * (1.0f / (real)nc);

  dt = dt / 1000000.0f;
  real idt = 1.0f / dt;
  int tau = ((int)( itau * idt) > 0) ? ((int)( itau * idt)) : 0;
  int w = (2 * tau) + 1;

  int number_of_semblances = 0;

  LOG(INFO, "Starting CMP execution");

  // Alloc memory
  iris::DMem *d_h, *d_gx, *d_gy, *d_sx, *d_sy, *d_scalco, *d_cdpsmpl;
  iris::DMem *d_c, *d_num, *d_stt, *d_str, *d_stk; // nc stts per sample
  iris::DMem *d_ctr; // ns Cs per cdp
  const size_t traces_bytes = ttraces * sizeof(real);
  d_gx = create_mem(traces_bytes);
  d_gy = create_mem(traces_bytes);
  d_sx = create_mem(traces_bytes);
  d_sy = create_mem(traces_bytes);
  d_scalco = create_mem(traces_bytes);
  d_cdpsmpl = create_mem(sizeof(real)*ntrs*ns);

  copy_to(d_gx    , h_gx    , traces_bytes);
  copy_to(d_gy    , h_gy    , traces_bytes);
  copy_to(d_sx    , h_sx    , traces_bytes);
  copy_to(d_sy    , h_sy    , traces_bytes);
  copy_to(d_scalco, h_scalco, traces_bytes);

  d_c = create_mem(sizeof(real)*nc);
  d_h = create_mem(sizeof(real)*ttraces);
  d_num = create_mem(sizeof(real)*ns*nc);
  d_stt = create_mem(sizeof(real)*ns*nc);
  d_ctr = create_mem(sizeof(int )*ncdps*ns);
  d_str = create_mem(sizeof(real)*ncdps*ns);
  d_stk = create_mem(sizeof(real)*ncdps*ns);

  h_ctr = (int*) malloc (sizeof(int )*ncdps*ns);
  h_str = (real*) malloc (sizeof(real)*ncdps*ns);
  h_stk = (real*) malloc (sizeof(real)*ncdps*ns);

  //
  // DEVICE REGION
  //
  
  auto beg = std::chrono::high_resolution_clock::now();

  // Evaluate Cs - linspace
  launch_init_c(nc, 1, d_c, inc, c0);

  // Evaluate halfoffset points in x and y coordinates
  launch_init_half(ttraces, 1, d_scalco, d_gx, d_gy, d_sx, d_sy, d_h);

  for(int cdp_id = 0; cdp_id < ncdps; cdp_id++) {
    int t_id0 = cdp_id > 0 ? ntraces_by_cdp_id[cdp_id-1] : 0;
    int t_idf = ntraces_by_cdp_id[cdp_id];
    int stride = t_idf - t_id0;

    copy_to(d_cdpsmpl, h_samples + t_id0*ns , sizeof(real)*stride*ns );

    // Compute semblances for each c for each sample
    launch_compute_semblances((ns*nc+NTHREADS-1)/NTHREADS, NTHREADS, 
        d_h, d_c, d_cdpsmpl, d_num, d_stt, t_id0, t_idf, idt, dt, tau, w, nc, ns);

    // Get max C for max semblance for each sample on this cdp
    launch_redux_semblances((ns+NTHREADS-1)/NTHREADS, NTHREADS, 
        d_num, d_stt, d_ctr, d_str, d_stk, nc, cdp_id, ns);

    number_of_semblances += stride;

#ifdef DEBUG
    std::cout << "Progress: " + std::to_string(cdp_id) + "/" + std::to_string(ncdps) << std::endl;
#endif
  }
  // Gets time at end of computation
  
  auto end = std::chrono::high_resolution_clock::now();

  // Copy results back to host
  d_ctr->fetch(h_ctr, sizeof(int ) * ncdps * ns);
  d_str->fetch(h_str, sizeof(real) * ncdps * ns);
  d_stk->fetch(h_stk, sizeof(real) * ncdps * ns);

  //
  // END DEVICE REGION
  //

  // Verify
  real* h_c   = (real*) malloc (sizeof(real)*nc      );
  real* h_h   = (real*) malloc (sizeof(real)*ttraces );
  real* h_num = (real*) malloc (sizeof(real)*ns*nc   );
  real* h_stt = (real*) malloc (sizeof(real)*ns*nc   );

  // reference results
   int* r_ctr = (int*) malloc (sizeof(int )*ncdps*ns);
  real* r_str = (real*) malloc (sizeof(real)*ncdps*ns);
  real* r_stk = (real*) malloc (sizeof(real)*ncdps*ns);

  h_init_c(nc, h_c, inc, c0);

  h_init_half(ttraces, h_scalco, h_gx, h_gy, h_sx, h_sy, h_h);

  for(int cdp_id = 0; cdp_id < ncdps; cdp_id++) {
    int t_id0 = cdp_id > 0 ? ntraces_by_cdp_id[cdp_id-1] : 0;
    int t_idf = ntraces_by_cdp_id[cdp_id];

    // Compute semblances for each c for each sample
    h_compute_semblances(
        h_h, h_c, h_samples+t_id0*ns, h_num, h_stt, t_id0, t_idf, idt, dt, tau, w, nc, ns);

    // Get max C for max semblance for each sample on this cdp
    h_redux_semblances(h_num, h_stt, r_ctr, r_str, r_stk, nc, cdp_id, ns);
#ifdef DEBUG
    std::cout << "Progress: " + std::to_string(cdp_id) + "/" + std::to_string(ncdps) << std::endl;
#endif
  }

  int err_ctr = 0, err_str = 0, err_stk = 0;
  for (int i = 0; i < ncdps*ns; i++) {
   if (r_ctr[i] != h_ctr[i]) err_ctr++;
   if (r_str[i] - h_str[i] > 1e-3) err_str++;
   if (r_stk[i] - h_stk[i] > 1e-3) err_stk++;
  }
  float err_ctr_rate = (float)err_ctr / (ncdps * ns);
  float err_str_rate = (float)err_str / (ncdps * ns); 
  float err_stk_rate = (float)err_stk / (ncdps * ns); 
  printf("Error rate: ctr=%e str=%e stk=%e\n",
         err_ctr_rate, err_str_rate, err_stk_rate);

  // Logs stats (semblance-traces per second)
  double time = std::chrono::duration_cast<std::chrono::duration<double>>(end - beg).count();
  double stps = (number_of_semblances / 1e9 ) * (ns * nc / time);
  std::string stats = "Giga semblances traces per second: " + std::to_string(stps);
  LOG(INFO, stats);

#ifdef SAVE
  // Delinearizes data and save it into a *.su file
  for(int i=0; i < ncdps; i++) {
    su_trace ctr_t = gather[i].traces()[0];
    su_trace str_t = gather[i].traces()[0];
    su_trace stk_t = gather[i].traces()[0];

    ctr_t.offset() = 0;
    ctr_t.sx() = ctr_t.gx() = (gather[i].traces()[0].sx() + gather[i].traces()[0].gx()) >> 1;
    ctr_t.sy() = ctr_t.gy() = (gather[i].traces()[0].sy() + gather[i].traces()[0].gy()) >> 1;

    for(int k=0; k < ns; k++) ctr_t.data()[k] = h_ctr[i*ns+k] < 0 ? 0.0f: (c0 + inc * h_ctr[i*ns+k]) / FACTOR;
    str_t.data().assign(h_str + i*ns, h_str + (i+1)*ns);
    stk_t.data().assign(h_stk + i*ns, h_stk + (i+1)*ns);

    ctr_t.fputtr(c_out);
    str_t.fputtr(s_out);
    stk_t.fputtr(stack);
  }
#endif

  release_mem(d_gx     );
  release_mem(d_gy     );
  release_mem(d_sx     );
  release_mem(d_sy     );
  release_mem(d_scalco );
  release_mem(d_cdpsmpl);
  release_mem(d_h      );
  release_mem(d_c      );
  release_mem(d_num    );
  release_mem(d_stt    );
  release_mem(d_ctr    );
  release_mem(d_str    );
  release_mem(d_stk    );

  if(const char *path=getenv("HECBENCH_VALIDATION_DUMP")) {
    FILE *f=fopen(path,"wb"); if(!f) exit(4);
    int meta[]={ncdps,ns};
    if(fwrite(meta,sizeof(int),2,f)!=2) exit(4);
    if(fwrite(h_ctr,sizeof(int),ncdps*ns,f)!=size_t(ncdps*ns)) exit(4);
    if(fwrite(h_str,sizeof(real),ncdps*ns,f)!=size_t(ncdps*ns)) exit(4);
    if(fwrite(h_stk,sizeof(real),ncdps*ns,f)!=size_t(ncdps*ns)) exit(4);
    if(fwrite(r_ctr,sizeof(int),ncdps*ns,f)!=size_t(ncdps*ns)) exit(4);
    if(fwrite(r_str,sizeof(real),ncdps*ns,f)!=size_t(ncdps*ns)) exit(4);
    if(fwrite(r_stk,sizeof(real),ncdps*ns,f)!=size_t(ncdps*ns)) exit(4);
    fclose(f);
  }

  free(h_ctr    );
  free(h_str    );
  free(h_stk    );
  free(h_h      );
  free(h_c      );
  free(h_num    );
  free(h_stt    );
  free(r_ctr    );
  free(r_str    );
  free(r_stk    );

  delete [] h_gx              ;
  delete [] h_gy              ;
  delete [] h_sx              ;
  delete [] h_sy              ;
  delete [] h_scalco          ;
  delete [] h_samples         ;
  delete [] ntraces_by_cdp_id ;

  platform.finalize();
  return EXIT_SUCCESS;
}
