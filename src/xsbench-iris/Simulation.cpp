#include "XSbench_header.h"
#include <iris/iris.hpp>
#include "validation.h"
unsigned long long
run_event_based_simulation(Inputs in, SimulationData SD,
                           int mype, double *kernel_time)
{

  ////////////////////////////////////////////////////////////////////////////////
  // SUMMARY: Simulation Data Structure Manifest for "SD" Object
  // Here we list all heap arrays (and lengths) in SD that would need to be
  // offloaded manually if using an accelerator with a seperate memory space
  ////////////////////////////////////////////////////////////////////////////////
  // int * num_nucs;                     // Length = length_num_nucs;
  // double * concs;                     // Length = length_concs
  // int * mats;                         // Length = length_mats
  // double * unionized_energy_array;    // Length = length_unionized_energy_array
  // int * index_grid;                   // Length = length_index_grid
  // NuclideGridPoint * nuclide_grid;    // Length = length_nuclide_grid
  //
  // Note: "unionized_energy_array" and "index_grid" can be of zero length
  //        depending on lookup method.
  //
  // Note: "Lengths" are given as the number of objects in the array, not the
  //       number of bytes.
  ////////////////////////////////////////////////////////////////////////////////

  if(mype==0) printf("Beginning event based simulation...\n");

  // Let's create an extra verification array to reduce manually later on
  if( mype == 0 )
     printf("Allocating an additional %.1lf MB of memory for verification arrays...\n",
            in.lookups * sizeof(int) /1024.0/1024.0);

  int * verification_h = (int *) malloc(in.lookups * sizeof(int));

  if(mype == 0 ) printf("Initializing device buffers and JIT compiling kernel...\n");

  ////////////////////////////////////////////////////////////////////////////////
  // Create Device Buffers
  ////////////////////////////////////////////////////////////////////////////////

  iris::DMem *verification_d = nullptr;
  iris::DMem *mats_d = nullptr;
  iris::DMem *num_nucs_d = nullptr;
  iris::DMem *concs_d = nullptr;
  iris::DMem *nuclide_grid_d = nullptr;

  //buffer<int, 1> num_nucs_d(SD.num_nucs,SD.length_num_nucs);
  num_nucs_d = new iris::DMem(SD.num_nucs, sizeof(int) * SD.length_num_nucs);

  //buffer<double, 1> concs_d(SD.concs, SD.length_concs);
  concs_d = new iris::DMem(SD.concs, sizeof(double) * SD.length_concs);

  //buffer<int, 1> mats_d(SD.mats, SD.length_mats);
  mats_d = new iris::DMem(SD.mats, sizeof(int) * SD.length_mats);

  //buffer<NuclideGridPoint, 1> nuclide_grid_d(SD.nuclide_grid, SD.length_nuclide_grid);
  nuclide_grid_d = new iris::DMem(SD.nuclide_grid, sizeof(NuclideGridPoint) * SD.length_nuclide_grid);

  //buffer<int, 1> verification_d(verification_h, in.lookups);
  verification_d = new iris::DMem(verification_h,sizeof(int)*in.lookups);

  // These two are a bit of a hack. Sometimes they are empty buffers (if using hash or nuclide
  // grid methods). OpenCL will throw an example when we try to create an empty buffer. So, we
  // will just allocate some memory for them and move them as normal. The rest of our code
  // won't actually use them if they aren't needed, so this is safe. Probably a cleaner way
  // of doing this.
  if( SD.length_unionized_energy_array == 0 )
  {
    SD.length_unionized_energy_array = 1;
    SD.unionized_energy_array = (double *) malloc(sizeof(double));
  }
  //buffer<double,1> unionized_energy_array_d(SD.unionized_energy_array, SD.length_unionized_energy_array);
  iris::DMem *unionized_energy_array_d = nullptr;
  unionized_energy_array_d = new iris::DMem(SD.unionized_energy_array, sizeof(double) * SD.length_unionized_energy_array);

  if( SD.length_index_grid == 0 )
  {
    SD.length_index_grid = 1;
    SD.index_grid = (int *) malloc(sizeof(int));
  }

  //buffer<int, 1> index_grid_d(SD.index_grid, (unsigned long long ) SD.length_index_grid);
  iris::DMem *index_grid_d = nullptr;
  index_grid_d = new iris::DMem(SD.index_grid, sizeof(int) * (unsigned long long)SD.length_index_grid);



  ////////////////////////////////////////////////////////////////////////////////
  // Define Device Kernel
  ////////////////////////////////////////////////////////////////////////////////

  double kstart = get_time();

  for (int i = 0; i < in.kernel_repeat; i++) {
    iris::Task task;
    void *p[]={num_nucs_d,concs_d,mats_d,nuclide_grid_d,verification_d,unionized_energy_array_d,index_grid_d,
      &in.lookups,&in.n_isotopes,&in.n_gridpoints,&in.grid_type,&in.hash_bins,&SD.max_num_nucs};
    int info[]={iris_r,iris_r,iris_r,iris_r,iris_w,iris_r,iris_r,sizeof(in.lookups),sizeof(in.n_isotopes),sizeof(in.n_gridpoints),sizeof(in.grid_type),sizeof(in.hash_bins),sizeof(SD.max_num_nucs)};
    size_t g[]={size_t((in.lookups+255)/256)*256},l[]={256};
    task.kernel("lookup",1,nullptr,g,l,13,p,info);
    if(i==in.kernel_repeat-1) task.flush_out(*verification_d);
    if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
  }


  double kstop = get_time();
  *kernel_time = (kstop - kstart) / in.kernel_repeat;

  export_verification("",verification_h,in.lookups);

  delete verification_d;
  delete mats_d;
  delete num_nucs_d;
  delete concs_d;
  delete nuclide_grid_d;
  delete unionized_energy_array_d;
  delete index_grid_d;

  // Host reduces the verification array
  unsigned long long verification_scalar = 0;
  for( int i = 0; i < in.lookups; i++ )
    verification_scalar += verification_h[i];

  if( SD.length_unionized_energy_array == 0 ) free(SD.unionized_energy_array);
  if( SD.length_index_grid == 0 ) free(SD.index_grid);
  free(verification_h);

  return verification_scalar;
}
