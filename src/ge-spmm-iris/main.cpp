#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <iris/iris.hpp>
#include <cmath>

#include "./util/mmio.hpp"
#include "./util/util.hpp"

// validate GPU results
#define VALIDATE



#define CLEANUP(s) \
  do { \
    printf("%s", s); \
    if (A_data) free(A_data); \
    if (A_indptr) free(A_indptr); \
    if (A_indices) free(A_indices); \
    if (B) free(B); \
    if (C) free(C); \
    if (golden) free(golden); \
    if (A_data_dev) delete A_data_dev; \
    if (A_indptr_dev) delete A_indptr_dev; \
    if (A_indices_dev) delete A_indices_dev; \
    if (B_dev) delete B_dev; \
    if (C_dev) delete C_dev; \
    fflush(stdout); \
  } while (0)

static void prove_backend() {
 int n=0; iris_device_count(&n); bool found=false;
 for(int i=0;i<n;++i) { int b=0,t=0; char name[256]={}; size_t z;
 iris_device_info(i,iris_backend,&b,&z); iris_device_info(i,iris_type,&t,&z); iris_device_info(i,iris_name,name,&z);
 printf("IRIS backend%d/type%d %s\n",b,t,name); if(b==3&&t==256) found=true; }
 if(!found) exit(2);
}

void spmmWrapper(int method,int tile_row,int A_nrows,int B_ncols,
 iris::DMem *A_rowPtr,iris::DMem *A_colInd,iris::DMem *A_val,iris::DMem *B,iris::DMem *C) {
 int width=32<<(method-1);
 size_t l[]={32,size_t(tile_row)},g[]={size_t((A_nrows+tile_row-1)/tile_row)*32,size_t((B_ncols+width-1)/width)*tile_row};
 iris::Task task; void *p[]={&A_nrows,&B_ncols,A_rowPtr,A_colInd,A_val,B,C};
 int info[]={sizeof(A_nrows),sizeof(B_ncols),iris_r,iris_r,iris_r,iris_r,iris_rw};
 const char *names[]={"spmm_test1","spmm_test2","spmm_test3","spmm_test4"};
 task.kernel(names[method-1],2,nullptr,g,l,7,p,info); task.flush_out(*C);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}


int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <matrix file> <tile row> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform; platform.init(&argc,&argv,1);
  prove_backend();
  int A_nrows, A_ncols, nnz, B_ncols;
  int max_ncols=256; // 256, 512

  std::vector<int> row_indices;
  std::vector<int> col_indices;
  std::vector<float> values;

  // Host allocate
  int* A_indptr = 0;
  int* A_indices = 0;
  float* A_data = 0;
  float* B = 0;
  float* C = 0;
  float* golden = 0;

  // Device allocate
  iris::DMem* A_data_dev = nullptr;
  iris::DMem* A_indices_dev = nullptr;
  iris::DMem* A_indptr_dev = nullptr;
  iris::DMem* B_dev = nullptr;
  iris::DMem* C_dev = nullptr;

  printf("reading data file ...\n");
  readMtx<float>(argv[1], row_indices, col_indices, values, A_nrows, A_ncols, nnz);

  const int tile_row = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  A_data = (float *)malloc(nnz*sizeof(A_data[0]));
  A_indptr = (int *)malloc((A_nrows+1)*sizeof(A_indptr[0]));
  A_indices = (int *)malloc(nnz*sizeof(A_indices[0]));
  B = (float *)malloc((max_ncols*A_ncols)*sizeof(B[0]));

  if ( !A_data || !A_indices || !A_indptr || !B ) {
    CLEANUP("Host malloc failed\n");
    return 1;
  }

#ifdef VALIDATE
  C = (float *)malloc((A_nrows*max_ncols)*sizeof(C[0]));
  golden = (float *)malloc((A_nrows*max_ncols)*sizeof(golden[0]));
  if (!C || !golden) {
    CLEANUP("Host malloc failed\n");
    return 1;
  }
#endif

  /* format conversation COO -> CSR */
  for (int i=0; i<A_nrows+1; i++) {
    A_indptr[i] = 0;
  }
  for (int n=0; n<nnz; n++) {
    int row = row_indices[n];
    if (row>=A_nrows) fprintf(stderr, "out of bound row\n");
    A_indptr[row+1]++;
  }
  for (int n=1; n<A_nrows+1; n++) {
    A_indptr[n] += A_indptr[n-1];
  }
  for (int n=0; n<nnz; n++) {
    int ptr = A_indptr[row_indices[n]];
    if (col_indices[n]>A_ncols) fprintf(stderr, "out of bound column\n");
    A_indices[ptr] = col_indices[n];
    // A_data[ptr] = values[n];
    A_data[ptr] = 1;
    ptr++;
    A_indptr[row_indices[n]]=ptr;
  }
  for (int n=A_nrows-1; n>0; n--) {
    A_indptr[n] = A_indptr[n-1];
  }
  A_indptr[0] = 0; // COO->CSR finish

  printf("read file ok. N=%d nnz=%d\n", A_nrows, nnz);

  /* random assign */
  srand(123);
  for (int i=0; i<max_ncols*A_ncols; i++)
    B[i] = float(rand()%100 - 50)/100;

#ifdef VALIDATE
  for (int i=0; i<A_nrows; i++) {
    for (int k=0; k<max_ncols; k++) {
      float acc = 0.f;
      for (int ptr=A_indptr[i]; ptr<A_indptr[i+1]; ptr++) {
        acc += A_data[ptr]*B[(max_ncols*A_indices[ptr]+k)];
      }
      golden[(max_ncols*i+k)] = acc;
    }
  }
#endif

  A_indptr_dev = new iris::DMem(A_indptr,(A_nrows+1)*sizeof(int));
  A_indices_dev = new iris::DMem(A_indices,nnz*sizeof(int));
  A_data_dev = new iris::DMem(A_data,nnz*sizeof(float));
  B_dev = new iris::DMem(B,max_ncols*A_ncols*sizeof(float));
  C_dev = new iris::DMem(C,max_ncols*A_nrows*sizeof(float));
  // execute spmm
  bool ok = true;
  for (B_ncols=256; B_ncols<=max_ncols; B_ncols *= 2) {
    NEXT_METHOD:
    for (int method=1; method<5; method++) {
      memset(C,0,A_nrows*B_ncols*sizeof(float)); C_dev->update(C);

      
      auto start = std::chrono::steady_clock::now();

      for (int i=0; i<repeat; i++)
        spmmWrapper(method, tile_row,  A_nrows, B_ncols, A_indptr_dev,
                    A_indices_dev, A_data_dev, B_dev, C_dev);

      
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel (method %d) execution time %f (us)\n", method, (time * 1e-3f) / repeat);

      // C is visible through task.flush_out.
      #ifdef VALIDATE
      for (int i=0; i<A_nrows; i++) 
        for (int j=0; j<B_ncols; j++) 
          if ( fabs((C[(i*B_ncols+j)] - golden[(i*B_ncols+j)])) > 1e-2 ) {
            printf("b_ncols %d kernel method %d: results mismatch %f %f\n",
                    B_ncols, method, C[(i*B_ncols+j)], golden[(i*B_ncols+j)]);
            ok = false;
            goto NEXT_METHOD;
          }
      #endif
    }
  }

  CLEANUP("");
  platform.finalize();
  printf("%s\n", ok ? "PASS" : "FAIL");
  
  return 0;
}
