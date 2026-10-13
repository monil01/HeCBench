#include "rsbench.h"
#include <iris/iris.hpp>
void run_event_based_simulation(Input in, SimulationData SD, unsigned long * vhash_result, double * kernel_time )
{
  printf("Beginning event based simulation...\n");

  // Let's create an extra verification array to reduce manually later on
  printf("Allocating an additional %.1lf MB of memory for verification arrays...\n", in.lookups * sizeof(int) /1024.0/1024.0);
  int * verification_host = (int *) malloc(in.lookups * sizeof(int));

  // Scope here is important, as when we exit this blocl we will automatically sync with device
  // to ensure all work is done and that we can read from verification_host array.
  
  // create a queue using the default device for the platform (cpu, gpu)

  printf("Initializing device buffers and JIT compiling kernel...\n");

  ////////////////////////////////////////////////////////////////////////////////
  // Create Device Buffers
  ////////////////////////////////////////////////////////////////////////////////
  iris::DMem *verification_d = nullptr;
  iris::DMem *mats_d = nullptr;
  iris::DMem *num_nucs_d = nullptr;
  iris::DMem *n_windows_d = nullptr;
  iris::DMem *concs_d = nullptr;
  iris::DMem *pseudo_K0RS_d = nullptr;
  iris::DMem *windows_d = nullptr;
  iris::DMem *poles_d = nullptr;

  // assign SYCL buffer to existing memory
  //buffer<int, 1> num_nucs_d(SD.num_nucs,SD.length_num_nucs);
  num_nucs_d = new iris::DMem(SD.num_nucs, sizeof(int) * SD.length_num_nucs);

  //buffer<double, 1> concs_d(SD.concs, SD.length_concs);
  concs_d = new iris::DMem(SD.concs, sizeof(double) * SD.length_concs);

  //buffer<int, 1> mats_d(SD.mats, SD.length_mats);
  mats_d = new iris::DMem(SD.mats, sizeof(int) * SD.length_mats);

  //buffer<int, 1> n_windows_d(SD.n_windows, SD.length_n_windows);
  n_windows_d = new iris::DMem(SD.n_windows, sizeof(int) * SD.length_n_windows);

  //buffer<Pole, 1> poles_d(SD.poles, SD.length_poles);
  poles_d = new iris::DMem(SD.poles, sizeof(Pole) * SD.length_poles);

  //buffer<Window, 1> windows_d(SD.windows, SD.length_windows);
  windows_d = new iris::DMem(SD.windows, sizeof(Window) * SD.length_windows);
  //buffer<double, 1> pseudo_K0RS_d(SD.pseudo_K0RS, SD.length_pseudo_K0RS);
  pseudo_K0RS_d = new iris::DMem(SD.pseudo_K0RS, sizeof(double) * SD.length_pseudo_K0RS);

  //buffer<int, 1> verification_d(verification_host, in.lookups);
  verification_d = new iris::DMem(verification_host, sizeof(int) * in.lookups);

  auto start = std::chrono::steady_clock::now();

  ////////////////////////////////////////////////////////////////////////////////
  // XS Lookup Simulation Loop
  ////////////////////////////////////////////////////////////////////////////////
  iris::Task task;
  void *p[]={num_nucs_d,concs_d,mats_d,verification_d,n_windows_d,pseudo_K0RS_d,windows_d,poles_d,
    &in.lookups,&in.doppler,&in.numL,&SD.max_num_windows,&SD.max_num_poles,&SD.max_num_nucs};
  int info[]={iris_r,iris_r,iris_r,iris_w,iris_r,iris_r,iris_r,iris_r,
    sizeof(int),sizeof(int),sizeof(int),sizeof(int),sizeof(int),sizeof(int)};
  size_t g[]={size_t((in.lookups+255)/256)*256},l[]={256};
  task.kernel("lookup",1,nullptr,g,l,14,p,info); task.flush_out(*verification_d);
  if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
  auto stop = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  printf("Kernel initialization, compilation, and execution took %.2lf seconds.\n", time * 1e-9);



  if(const char *path=getenv("HECBENCH_VALIDATION_DUMP")) {
    FILE *f=fopen(path,"wb");
    if(!f || fwrite(verification_host,sizeof(int),in.lookups,f)!=size_t(in.lookups)) exit(4);
    fclose(f);
  }
  delete verification_d;
  delete mats_d;
  delete num_nucs_d;
  delete concs_d;
  delete n_windows_d;
  delete windows_d;
  delete poles_d;
  delete pseudo_K0RS_d;

  // Host reduces the verification array
  unsigned long long verification_scalar = 0;
  for( int i = 0; i < in.lookups; i++ )
    verification_scalar += verification_host[i];

  *vhash_result = verification_scalar;
  *kernel_time = time * 1e-9;
}
