//====================================================================================================100
//    UPDATE
//====================================================================================================100

//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//    2009.12 Lukasz G. Szafaryn
//    -- reading from image, command line inputs
//    2010.01 Lukasz G. Szafaryn
//    --comments


#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <iris/iris.hpp>
#include <vector>
struct dim3 { unsigned x=1,y=1,z=1; };

#include "main.h"
#include "graphics.c"
#include "resize.c"

int main(int argc, char *argv []) {
  iris::Platform platform;
  if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) return 1;
  bool gpu=false;
  for(int d=0;d<platform.ndevs();d++) {
    int backend=0,type=0; char name[256]={}; size_t bytes=0;
    iris_device_info(d,iris_backend,&backend,&bytes); iris_device_info(d,iris_type,&type,&bytes); iris_device_info(d,iris_name,name,&bytes);
    printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name); gpu |= backend==3 && type==256;
  }
  if(!gpu) return 1;

  // times
  std::chrono::steady_clock::time_point time[13];

  time[0] = std::chrono::steady_clock::now();

  // inputs image, input paramenters
  fp* image_ori;  // original input image
  int image_ori_rows;
  int image_ori_cols;
  long image_ori_elem;

  // inputs image, input paramenters
  fp* image; // input image
  int Nr,Nc; // IMAGE nbr of rows/cols/elements
  long Ne;

  // algorithm parameters
  int niter; // nbr of iterations
  fp lambda; // update step size

    // size of IMAGE
  int r1,r2,c1,c2; // row/col coordinates of uniform ROI
  long NeROI; // ROI nbr of elements

  // surrounding pixel indicies
  int *iN,*iS,*jE,*jW;    

  // counters
  int iter; // primary loop
  long i,j; // image row/col

  // memory sizes
  int mem_size_i;
  int mem_size_j;
  int mem_size_single;

  //================================================================================80
  //   GPU VARIABLES
  //================================================================================80

  // CUDA kernel execution parameters
  dim3 threads;
  int blocks_x;
  dim3 blocks;
  dim3 blocks2;

  // matrix memory size
  int mem_size;

  // HOST
  int no;
  int mul;
  fp total;
  fp total2;
  fp meanROI;
  fp meanROI2;
  fp varROI;
  fp q0sqr;

  time[1] = std::chrono::steady_clock::now();

  //================================================================================80
  //   GET INPUT PARAMETERS
  //================================================================================80

  if(argc != 5){
    printf("Usage: %s <repeat> <lambda> <number of rows> <number of columns>\n", argv[0]);
    return 1;
  }
  else{
    niter = atoi(argv[1]);
    lambda = atof(argv[2]);
    Nr = atoi(argv[3]);
    Nc = atoi(argv[4]);
  }

  time[2] = std::chrono::steady_clock::now();

  //================================================================================80
  //   READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
  //================================================================================80

    // read original image
  image_ori_rows = 502;
  image_ori_cols = 458;
  image_ori_elem = image_ori_rows * image_ori_cols;
  image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

  const char* input_image_path = "../data/srad/image.pgm";
  if ( !read_graphics(input_image_path, image_ori, image_ori_rows, image_ori_cols, 1) ) { 
    printf("ERROR: failed to read input image at %s\n", input_image_path);
    if (image_ori != NULL) free(image_ori);
    return -1;
  }

  time[3] = std::chrono::steady_clock::now();

  //================================================================================80
  //   RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
  //================================================================================80

  Ne = Nr*Nc;

  image = (fp*)malloc(sizeof(fp) * Ne);

  resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

  time[4] = std::chrono::steady_clock::now();

  //================================================================================80
  //   SETUP
  //================================================================================80

  r1     = 0;      // top row index of ROI
  r2     = Nr - 1; // bottom row index of ROI
  c1     = 0;      // left column index of ROI
  c2     = Nc - 1; // right column index of ROI

  // ROI image size
  NeROI = (r2-r1+1)*(c2-c1+1); // number of elements in ROI, ROI size

  // allocate variables for surrounding pixels
  mem_size_i = sizeof(int) * Nr;
  iN = (int *)malloc(mem_size_i) ;  // north surrounding element
  iS = (int *)malloc(mem_size_i) ;  // south surrounding element
  mem_size_j = sizeof(int) * Nc;    
  jW = (int *)malloc(mem_size_j) ;  // west surrounding element
  jE = (int *)malloc(mem_size_j) ;  // east surrounding element

  // N/S/W/E indices of surrounding pixels (every element of IMAGE)
  for (i=0; i<Nr; i++) {
    iN[i] = i-1;                            // holds index of IMAGE row above
    iS[i] = i+1;                            // holds index of IMAGE row below
  }
  for (j=0; j<Nc; j++) {
    jW[j] = j-1;                            // holds index of IMAGE column on the left
    jE[j] = j+1;                            // holds index of IMAGE column on the right
  }

  // N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
  iN[0]    = 0;                             // changes IMAGE top row index from -1 to 0
  iS[Nr-1] = Nr-1;                          // changes IMAGE bottom row index from Nr to Nr-1 
  jW[0]    = 0;                             // changes IMAGE leftmost column index from -1 to 0
  jE[Nc-1] = Nc-1;                          // changes IMAGE rightmost column index from Nc to Nc-1

  //================================================================================80
  //   GPU SETUP
  //================================================================================80

  // allocate memory for entire IMAGE on DEVICE
  mem_size = sizeof(fp) * Ne; // get the size of float representation of input IMAGE
  iris::DMem d_I(image,mem_size),d_iN(iN,mem_size_i),d_iS(iS,mem_size_i),d_jE(jE,mem_size_j),d_jW(jW,mem_size_j);
  std::vector<fp> sums(Ne),sums2(Ne),dn(Ne),ds(Ne),dw(Ne),de(Ne),c(Ne);
  iris::DMem d_sums(sums.data(),mem_size),d_sums2(sums2.data(),mem_size),d_dN(dn.data(),mem_size),d_dS(ds.data(),mem_size),d_dW(dw.data(),mem_size),d_dE(de.data(),mem_size),d_c(c.data(),mem_size);

  //checkCUDAError("setup");

  //================================================================================80
  //   KERNEL EXECUTION PARAMETERS
  //================================================================================80

  // all kernels operating on entire matrix
  threads.x = NUMBER_THREADS;        // define the number of threads in the block
  threads.y = 1;
  blocks_x = Ne/threads.x;
  if (Ne % threads.x != 0){          // compensate for division remainder above by adding one grid
    blocks_x = blocks_x + 1;                                  
  }
  blocks.x = blocks_x;               // define the number of blocks in the grid
  blocks.y = 1;

  time[5] = std::chrono::steady_clock::now();

  //================================================================================80
  //   COPY INPUT TO GPU
  //================================================================================80

  d_I.update(image);

  time[6] = std::chrono::steady_clock::now();

  //================================================================================80
  //   SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
  //================================================================================80

  {
      iris::Task task;
      size_t gws[]={size_t(blocks.x)*threads.x},lws[]={threads.x};
      void* args[]={&Ne,&d_I};
      int info[]={sizeof(Ne),iris_rw};
      if(task.kernel("extract",1,nullptr,gws,lws,2,args,info)!=IRIS_SUCCESS) return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 2;
    }

  //checkCUDAError("extract");


  time[7] = std::chrono::steady_clock::now();

  //================================================================================80
  //   COMPUTATION
  //================================================================================80

  // printf("iterations: ");

  // execute main loop
  for (iter=0; iter<niter; iter++){  // do for the number of iterations input parameter

  // printf("%d ", iter);
  // fflush(NULL);

    // execute square kernel
    {
      iris::Task task;
      size_t gws[]={size_t(blocks.x)*threads.x},lws[]={threads.x};
      void* args[]={&Ne,&d_I,&d_sums,&d_sums2};
      int info[]={sizeof(Ne),iris_rw,iris_rw,iris_rw};
      if(task.kernel("prepare",1,nullptr,gws,lws,4,args,info)!=IRIS_SUCCESS) return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 2;
    }

    //checkCUDAError("prepare");

    // performs subsequent reductions of sums
    blocks2.x = blocks.x;            // original number of blocks
    blocks2.y = blocks.y;                        
    no = Ne;                         // original number of sum elements
    mul = 1;                         // original multiplier

    while(blocks2.x != 0){

      //checkCUDAError("before reduce");

      // run kernel
      {
      iris::Task task;
      size_t gws[]={size_t(blocks2.x)*threads.x},lws[]={threads.x};
      void* args[]={&Ne,&no,&mul,&d_sums,&d_sums2};
      int info[]={sizeof(Ne),sizeof(no),sizeof(mul),iris_rw,iris_rw};
      if(task.kernel("reduce",1,nullptr,gws,lws,5,args,info)!=IRIS_SUCCESS) return 2;
      task.flush_out(d_sums); task.flush_out(d_sums2);
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 2;
    }
      //checkCUDAError("reduce");

      // update execution parameters
      no = blocks2.x;                // get current number of elements
      if(blocks2.x == 1){
        blocks2.x = 0;
      }
      else{
        mul = mul * NUMBER_THREADS;  // update the increment
        blocks_x = blocks2.x/threads.x;   // number of blocks
        if (blocks2.x % threads.x != 0){  // compensate for division remainder above by adding one grid
          blocks_x = blocks_x + 1;
        }
        blocks2.x = blocks_x;
        blocks2.y = 1;
      }

      //checkCUDAError("after reduce");
    }

    //checkCUDAError("before copy sum");

    // copy total sums to device
    mem_size_single = sizeof(fp) * 1;
    total=sums[0];
    total2=sums2[0];

    //checkCUDAError("copy sum");

    // calculate statistics
    meanROI  = total / fp(NeROI);             // gets mean (average) value of element in ROI
    meanROI2 = meanROI * meanROI;             //
    varROI = (total2 / fp(NeROI)) - meanROI2; // gets variance of ROI                
    q0sqr = varROI / meanROI2;                // gets standard deviation of ROI

    // execute srad kernel
    {
      iris::Task task;
      size_t gws[]={size_t(blocks.x)*threads.x},lws[]={threads.x};
      void* args[]={&lambda,&Nr,&Nc,&Ne,&d_iN,&d_iS,&d_jE,&d_jW,&d_dN,&d_dS,&d_dW,&d_dE,&q0sqr,&d_c,&d_I};
      int info[]={sizeof(lambda),sizeof(Nr),sizeof(Nc),sizeof(Ne),iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(q0sqr),iris_rw,iris_rw};
      if(task.kernel("srad",1,nullptr,gws,lws,15,args,info)!=IRIS_SUCCESS) return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 2;
    }                  // output image

    //checkCUDAError("srad");

    // execute srad2 kernel
    {
      iris::Task task;
      size_t gws[]={size_t(blocks.x)*threads.x},lws[]={threads.x};
      void* args[]={&lambda,&Nr,&Nc,&Ne,&d_iN,&d_iS,&d_jE,&d_jW,&d_dN,&d_dS,&d_dW,&d_dE,&d_c,&d_I};
      int info[]={sizeof(lambda),sizeof(Nr),sizeof(Nc),sizeof(Ne),iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw};
      if(task.kernel("srad2",1,nullptr,gws,lws,14,args,info)!=IRIS_SUCCESS) return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 2;
    }                  // output image

    //checkCUDAError("srad2");
  }


  // printf("\n");

  time[8] = std::chrono::steady_clock::now();

  //================================================================================80
  //   SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
  //================================================================================80

  {
      iris::Task task;
      size_t gws[]={size_t(blocks.x)*threads.x},lws[]={threads.x};
      void* args[]={&Ne,&d_I};
      int info[]={sizeof(Ne),iris_rw};
      if(task.kernel("compress",1,nullptr,gws,lws,2,args,info)!=IRIS_SUCCESS) return 2;
      task.flush_out(d_I);
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 2;
    }

  //checkCUDAError("compress");


  time[9] = std::chrono::steady_clock::now();

  //================================================================================80
  //   COPY RESULTS BACK TO CPU
  //================================================================================80

  if(const char* path=getenv("HECBENCH_ARTIFACT")) {
    FILE* out=fopen(path,"wb"); if(!out) return 3;
    fwrite(image,sizeof(fp),Ne,out); fclose(out);
  }

  //checkCUDAError("copy back");

  time[10] = std::chrono::steady_clock::now();

  //================================================================================80
  //   WRITE IMAGE AFTER PROCESSING
  //================================================================================80

  write_graphics(
          "image_out.pgm",
          image,
          Nr,
          Nc,
          1,
          255);

  time[11] = std::chrono::steady_clock::now();

  //================================================================================80
  //  DEALLOCATE
  //================================================================================80

  free(image_ori);
  free(image);
  free(iN); 
  free(iS); 
  free(jW); 
  free(jE);













  time[12] = std::chrono::steady_clock::now();

  //================================================================================80
  //  DISPLAY TIMING
  //================================================================================80

  printf("Time spent in different stages of the application:\n");
  long etime[13];
  for (int i = 0; i <= 11; i++) {
    etime[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(time[i+1] - time[i]).count();
  }
  etime[12] = std::chrono::duration_cast<std::chrono::nanoseconds>(time[12] - time[0]).count();

  printf("%15.12f s, %15.12f %% : SETUP VARIABLES\n",
         etime[0] * 1e-9, etime[0] * 100.0 / etime[12]);
  printf("%15.12f s, %15.12f %% : READ COMMAND LINE PARAMETERS\n",
         etime[1] * 1e-9, etime[1] * 100.0 / etime[12]);
  printf("%15.12f s, %15.12f %% : READ IMAGE FROM FILE\n",
         etime[2] * 1e-9, etime[2] * 100.0 / etime[12]);
  printf("%15.12f s, %15.12f %% : RESIZE IMAGE\n", 
         etime[3] * 1e-9, etime[3] * 100.0 / etime[12]);
  printf("%15.12f s, %15.12f %% : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n",
         etime[4] * 1e-9, etime[4] * 100.0 / etime[12]);
  printf("%15.12f s, %15.12f %% : COPY DATA TO CPU->GPU\n",
         etime[5] * 1e-9, etime[5] * 100.0 / etime[12]);
  printf("%15.12f s, %15.12f %% : EXTRACT IMAGE\n", 
         etime[6] * 1e-9, etime[6] * 100.0 / etime[12]);
  printf("%15.12f s, %15.12f %% : COMPUTE (%d iterations)\n", 
         etime[7] * 1e-9, etime[7] * 100.0 / etime[12], niter);
  printf("%15.12f s, %15.12f %% : COMPRESS IMAGE\n", 
         etime[8] * 1e-9, etime[8] * 100.0 / etime[12]);
  printf("%15.12f s, %15.12f %% : COPY DATA TO GPU->CPU\n", 
         etime[9] * 1e-9, etime[9] * 100.0 / etime[12]);
  printf("%15.12f s, %15.12f %% : SAVE IMAGE INTO FILE\n", 
         etime[10] * 1e-9, etime[10] * 100.0 / etime[12]);
  printf("%15.12f s, %15.12f %% : FREE MEMORY\n", 
         etime[11] * 1e-9, etime[11] * 100.0 / etime[12]);
  printf("Total time:\n");
  printf("%.12f s\n", etime[12] * 1e-9);

  int errors=platform.error_count(); platform.finalize();
  return errors ? 1 : 0;
}

