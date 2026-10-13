/***********************************************
  streamcluster_cl.h
  : parallelized code of streamcluster

  - original code from PARSEC Benchmark Suite
  - parallelization with OpenCL API has been applied by
  Jianbin Fang - j.fang@tudelft.nl
  Delft University of Technology
  Faculty of Electrical Engineering, Mathematics and Computer Science
  Department of Software Technology
  Parallel and Distributed Systems Group
  on 15/03/2010
 ***********************************************/

#include <iris/iris.hpp>
#include "point_struct.h"

static void prove_backend() {
  int count = 0; iris_device_count(&count);
  bool found = false;
  for (int i=0; i<count; ++i) {
    int backend=0, type=0; char name[256]={}; size_t size;
    iris_device_info(i, iris_backend, &backend, &size);
    iris_device_info(i, iris_type, &type, &size);
    iris_device_info(i, iris_name, name, &size);
    printf("IRIS device %d backend%d/type%d %s\\n", i, backend, type, name);
    if (backend==3 && type==256) found=true;
  }
  if (!found) { fprintf(stderr,"No HIP AMD GPU available\\n"); exit(2); }
}

#define THREADS_PER_BLOCK 256
#define MAXBLOCKS 65536





/* host memory analogous to device memory. These memories are allocated in the function,
 * but they are freed in the streamcluster.cpp. We cannot free them in the function as
 * the funtion is called repeatedly in streamcluster.cpp. */
float *work_mem_h;
float *coord_h;
float *gl_lower;
Point_Struct *p_h;

// device memory
iris::DMem *work_mem_d, *coord_d, *center_table_d, *switch_membership_d, *p_d;
static char *switch_h;
static int *table_h;
static size_t capacity;

static int c;      // counters

float pgain( long x, Points *points, float z, long int *numcenters,
             int kmax, bool *is_center, int *center_table, char *switch_membership,
             long *serial, long *cpu_gpu_memcpy, long *memcpy_back,
             long *gpu_malloc, long *kernel_time) {

  float gl_cost = 0;
  try{
#ifdef PROFILE_TMP
    auto t1 = std::chrono::steady_clock::now();
#endif
    int K = *numcenters ;   // number of centers
    int num = points->num;  // number of points
    int dim = points->dim;  // number of dimensions
    kmax++;
    /***** build center index table 1*****/
    int count = 0;
    for( int i=0; i<num; i++){
      if( is_center[i] )
        center_table[i] = count++;
    }

#ifdef PROFILE_TMP
    auto t2 = std::chrono::steady_clock::now();
    *serial += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
#endif

    /***** initial memory allocation and preparation for transfer : execute once *****/
    if( c == 0 ) {
#ifdef PROFILE_TMP
      auto t3 = std::chrono::steady_clock::now();
#endif
      coord_h = (float*) malloc( num * dim * sizeof(float)); // coordinates (host)
      gl_lower = (float*) malloc( kmax * sizeof(float) );
      work_mem_h = (float*) malloc (kmax*num*sizeof(float));
      p_h = (Point_Struct*) malloc (num*sizeof(Point_Struct));

      // prepare mapping for point coordinates
      for(int i=0; i<dim; i++){
        for(int j=0; j<num; j++)
          coord_h[ (num*i)+j ] = points->p[j].coord[i];
      }
#ifdef PROFILE_TMP
      auto t4 = std::chrono::steady_clock::now();
      *serial += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
#endif
      capacity = num;
      switch_h = (char*)calloc(num, sizeof(char));
      table_h = (int*)calloc(num, sizeof(int));
      work_mem_d = new iris::DMem(work_mem_h, sizeof(float)*kmax*num);
      center_table_d = new iris::DMem(table_h, sizeof(int)*num);
      switch_membership_d = new iris::DMem(switch_h, sizeof(char)*num);
      coord_d = new iris::DMem(coord_h, sizeof(float)*dim*num);
      p_d = new iris::DMem(p_h, sizeof(Point_Struct)*num);

#ifdef PROFILE_TMP

      auto t6 = std::chrono::steady_clock::now();
      *cpu_gpu_memcpy += std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t4).count();
#endif
    }    // first iteration

#ifdef PROFILE_TMP
    auto t100 = std::chrono::steady_clock::now();
#endif

    for(int i=0; i<num; i++){
      p_h[i].weight = ((points->p)[i]).weight;
      p_h[i].assign = ((points->p)[i]).assign;
      p_h[i].cost = ((points->p)[i]).cost;  
    }

#ifdef PROFILE_TMP
    auto t101 = std::chrono::steady_clock::now();
    *serial += std::chrono::duration_cast<std::chrono::nanoseconds>(t101 - t100).count();
#endif
#ifdef PROFILE_TMP
    auto t7 = std::chrono::steady_clock::now();
#endif
    /***** memory transfer from host to device *****/
    memcpy(table_h, center_table, sizeof(int)*num);
    center_table_d->update(table_h);
    p_d->update(p_h);

#ifdef PROFILE_TMP

    auto t8 = std::chrono::steady_clock::now();
    *cpu_gpu_memcpy += std::chrono::duration_cast<std::chrono::nanoseconds>(t8 - t7).count();
#endif

    /***** kernel execution *****/
    /* Determine the number of thread blocks in the x- and y-dimension */
    size_t smSize = dim;

#ifdef PROFILE_TMP
    auto t9 = std::chrono::steady_clock::now();
#endif

    memset(switch_h, 0, capacity*sizeof(char));
    memset(work_mem_h, 0, num*(K+1)*sizeof(float));
    switch_membership_d->update(switch_h);
    work_mem_d->update(work_mem_h);

    int work_group_size = THREADS_PER_BLOCK;
    int work_items = num;
    if(work_items%work_group_size != 0)  //process situations that work_items cannot be divided by work_group_size
      work_items = work_items + (work_group_size-(work_items%work_group_size));

    iris::Task task;
    void *params[] = {p_d, coord_d, work_mem_d, center_table_d, switch_membership_d, &num, &dim, &x, &K};
    int info[] = {iris_r, iris_r, iris_rw, iris_r, iris_rw, sizeof(num), sizeof(dim), sizeof(x), sizeof(K)};
    size_t global[] = {static_cast<size_t>(work_items)}, local[] = {THREADS_PER_BLOCK};
    task.kernel("compute_cost", 1, nullptr, global, local, 9, params, info);
    task.flush_out(*switch_membership_d);
    task.flush_out(*work_mem_d);
    if (task.submit(iris_gpu, nullptr, 1) != IRIS_SUCCESS) exit(3);

#ifdef PROFILE_TMP

    auto t10 = std::chrono::steady_clock::now();
    *kernel_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t10 - t9).count();
#endif

    /***** copy back to host for CPU side work *****/
    memcpy(switch_membership, switch_h, num*sizeof(char));


#ifdef PROFILE_TMP
    auto t11 = std::chrono::steady_clock::now();
    *memcpy_back += std::chrono::duration_cast<std::chrono::nanoseconds>(t11 - t10).count();
#endif

    /****** cpu side work *****/
    int numclose = 0;
    gl_cost = z;

    /* compute the number of centers to close if we are to open i */
    for(int i=0; i < num; i++){
      if( is_center[i] ) {
        float low = z;
        //printf("i=%d  ", i);
        for( int j = 0; j < num; j++ )
          low += work_mem_h[ j*(K+1) + center_table[i] ];
        //printf("low=%f\n", low);    
        gl_lower[center_table[i]] = low;

        if ( low > 0 ) {
          numclose++;        
          work_mem_h[i*(K+1)+K] -= low;
        }
      }
      gl_cost += work_mem_h[i*(K+1)+K];
    }

    /* if opening a center at x saves cost (i.e. cost is negative) do so
       otherwise, do nothing */
    if ( gl_cost < 0 ) {
      for(int i=0; i<num; i++){
        bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
        if ( (switch_membership[i]=='1') || close_center ) {
          points->p[i].cost = points->p[i].weight * dist(points->p[i], points->p[x], points->dim);
          points->p[i].assign = x;
        }
      }

      for(int i=0; i<num; i++){
        if( is_center[i] && gl_lower[center_table[i]] > 0 )
          is_center[i] = false;
      }

      is_center[x] = true;
      *numcenters = *numcenters +1 - numclose;
    }
    else
      gl_cost = 0;

#ifdef PROFILE_TMP
    auto t12 = std::chrono::steady_clock::now();
    *serial += std::chrono::duration_cast<std::chrono::nanoseconds>(t12 - t11).count();
#endif
    c++;
  }
  catch(string msg){
    printf("--cambine:%s\n", msg.c_str());
    exit(-1);    
  }
  catch(...){
    printf("--cambine: unknow reasons in pgain\n");
  }

#ifdef DEBUG
  FILE *fp = fopen("data_debug.txt", "a");
  fprintf(fp,"%d, %f\n", c, gl_cost);
  fclose(fp);
#endif
  return -gl_cost;
}
