#include <iris/iris.hpp>
#include <stdexcept>
static void check(int rc){if(rc!=IRIS_SUCCESS)throw std::runtime_error("IRIS API failure");}
template<class T> static void* arg(const T& value){return const_cast<T*>(&value);}
static int hip_device(){
  int n=0;check(iris_device_count(&n));
  for(int d=0;d<n;++d){int backend=0,type=0;char name[256]={};
    check(iris_device_info(d,iris_backend,&backend,nullptr));check(iris_device_info(d,iris_type,&type,nullptr));
    check(iris_device_info(d,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",d,name,backend,type);
    if(backend==iris_hip&&(type&iris_gpu))return d;
  }
  throw std::runtime_error("no IRIS HIP GPU");
}

#include <vector>
#include <algorithm>
#include <chrono>
#include "main.h"                // (in main directory)            needed to recognized input parameters
#include "util/avi/avilib.h"          // (in directory)              needed by avi functions
#include "util/avi/avimod.h"          // (in directory)              needed by avi functions

// CUDA kernel
static_assert(sizeof(params_common)<=1024,"IRIS scalar argument storage");

uint64_t
kernel_gpu_wrapper(  params_common common,
    int* endoRow,
    int* endoCol,
    int* tEndoRowLoc,
    int* tEndoColLoc,
    int* epiRow,
    int* epiCol,
    int* tEpiRowLoc,
    int* tEpiColLoc,
    avi_t* frames)
{

  // common
  //printf("tSize is %d, sSize is %d\n", common.tSize, common.sSize);
  common.in_rows = common.tSize + 1 + common.tSize;
  common.in_cols = common.in_rows;
  common.in_elem = common.in_rows * common.in_cols;
  common.in_mem = sizeof(fp) * common.in_elem;

  //==================================================50
  // endo points templates
  //==================================================50

  std::vector<fp> h_d_endoT((common.in_mem * common.endoPoints)/sizeof(fp),0);
  iris::DMem d_endoT(h_d_endoT.data(),common.in_mem * common.endoPoints);
  //printf("%d\n", common.in_elem * common.endoPoints);

  //==================================================50
  // epi points templates
  //==================================================50

  std::vector<fp> h_d_epiT((common.in_mem * common.epiPoints)/sizeof(fp),0);
  iris::DMem d_epiT(h_d_epiT.data(),common.in_mem * common.epiPoints);

  //====================================================================================================100
  //   AREA AROUND POINT    FROM  FRAME  (LOCAL)
  //====================================================================================================100

  // common
  common.in2_rows = common.sSize + 1 + common.sSize;
  common.in2_cols = common.in2_rows;
  common.in2_elem = common.in2_rows * common.in2_cols;
  common.in2_mem = sizeof(fp) * common.in2_elem;

  std::vector<fp> h_d_in2((common.in2_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in2(h_d_in2.data(),common.in2_mem * common.allPoints);
  //printf("%d\n", common.in2_elem * common.allPoints);

  //====================================================================================================100
  //   CONVOLUTION  (LOCAL)
  //====================================================================================================100

  // common
  common.conv_rows = common.in_rows + common.in2_rows - 1;                        // number of rows in I
  common.conv_cols = common.in_cols + common.in2_cols - 1;                        // number of columns in I
  common.conv_elem = common.conv_rows * common.conv_cols;                          // number of elements
  common.conv_mem = sizeof(fp) * common.conv_elem;
  common.ioffset = 0;
  common.joffset = 0;

  // unique
  std::vector<fp> h_d_conv((common.conv_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_conv(h_d_conv.data(),common.conv_mem * common.allPoints);

  //====================================================================================================100
  //   CUMULATIVE SUM  (LOCAL)
  //====================================================================================================100

  //==================================================50
  //   PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
  //==================================================50

  // common
  common.in2_pad_add_rows = common.in_rows;
  common.in2_pad_add_cols = common.in_cols;

  common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
  common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
  common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
  common.in2_pad_cumv_mem = sizeof(fp) * common.in2_pad_cumv_elem;

  // unique
  //buffer<fp,1> d_in2_pad_cumv(common.in2_pad_cumv_elem * common.allPoints);
  //printf("%d\n", common.in2_pad_cumv_elem * common.allPoints);
  std::vector<fp> h_d_in2_pad_cumv((common.in2_pad_cumv_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in2_pad_cumv(h_d_in2_pad_cumv.data(),common.in2_pad_cumv_mem * common.allPoints);

  //==================================================50
  //   SELECTION
  //==================================================50

  // common
  common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;                          // (1 to n+1)
  common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
  common.in2_pad_cumv_sel_collow = 1;
  common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
  common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
  common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
  common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
  common.in2_pad_cumv_sel_mem = sizeof(fp) * common.in2_pad_cumv_sel_elem;

  // unique
  //buffer<fp,1> d_in2_pad_cumv_sel(common.in2_pad_cumv_sel_elem * common.allPoints);
  //printf("%d\n", common.in2_pad_cumv_sel_elem * common.allPoints);
  std::vector<fp> h_d_in2_pad_cumv_sel((common.in2_pad_cumv_sel_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in2_pad_cumv_sel(h_d_in2_pad_cumv_sel.data(),common.in2_pad_cumv_sel_mem * common.allPoints);

  //==================================================50
  //   SELECTION  2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
  //==================================================50

  // common
  common.in2_pad_cumv_sel2_rowlow = 1;
  common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
  common.in2_pad_cumv_sel2_collow = 1;
  common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
  common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
  common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
  common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
  common.in2_sub_cumh_mem = sizeof(fp) * common.in2_sub_cumh_elem;

  // unique
  //buffer<fp,1> d_in2_sub_cumh(common.in2_sub_cumh_elem * common.allPoints);
  //printf("%d\n", common.in2_sub_cumh_elem * common.allPoints);
  std::vector<fp> h_d_in2_sub_cumh((common.in2_sub_cumh_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in2_sub_cumh(h_d_in2_sub_cumh.data(),common.in2_sub_cumh_mem * common.allPoints);

  //==================================================50
  //   SELECTION
  //==================================================50

  // common
  common.in2_sub_cumh_sel_rowlow = 1;
  common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
  common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
  common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
  common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
  common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
  common.in2_sub_cumh_sel_mem = sizeof(fp) * common.in2_sub_cumh_sel_elem;

  // unique
  //buffer<fp,1> d_in2_sub_cumh_sel(common.in2_sub_cumh_sel_elem * common.allPoints);
  //printf("%d\n", common.in2_sub_cumh_sel_elem * common.allPoints);
  std::vector<fp> h_d_in2_sub_cumh_sel((common.in2_sub_cumh_sel_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in2_sub_cumh_sel(h_d_in2_sub_cumh_sel.data(),common.in2_sub_cumh_sel_mem * common.allPoints);

  //==================================================50
  //  SELECTION 2, SUBTRACTION
  //==================================================50

  // common
  common.in2_sub_cumh_sel2_rowlow = 1;
  common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel2_collow = 1;
  common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
  common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
  common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
  common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
  common.in2_sub2_mem = sizeof(fp) * common.in2_sub2_elem;

  // unique
  //buffer<fp,1> d_in2_sub2(common.in2_sub2_elem * common.allPoints);
  //printf("%d\n", common.in2_sub2_elem * common.allPoints);
  std::vector<fp> h_d_in2_sub2((common.in2_sub2_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in2_sub2(h_d_in2_sub2.data(),common.in2_sub2_mem * common.allPoints);

  //====================================================================================================100
  //  CUMULATIVE SUM 2  (LOCAL)
  //====================================================================================================100

  //==================================================50
  //  MULTIPLICATION
  //==================================================50

  // common
  common.in2_sqr_rows = common.in2_rows;
  common.in2_sqr_cols = common.in2_cols;
  common.in2_sqr_elem = common.in2_elem;
  common.in2_sqr_mem = common.in2_mem;

  // unique
  //buffer<fp,1> d_in2_sqr(common.in2_elem * common.allPoints);
  //printf("%d\n", common.in2_elem * common.allPoints);
  std::vector<fp> h_d_in2_sqr((common.in2_sqr_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in2_sqr(h_d_in2_sqr.data(),common.in2_sqr_mem * common.allPoints);

  //==================================================50
  //  SELECTION 2, SUBTRACTION
  //==================================================50

  // common
  common.in2_sqr_sub2_rows = common.in2_sub2_rows;
  common.in2_sqr_sub2_cols = common.in2_sub2_cols;
  common.in2_sqr_sub2_elem = common.in2_sub2_elem;
  common.in2_sqr_sub2_mem = common.in2_sub2_mem;

  // unique
  //buffer<fp,1> d_in2_sqr_sub2(common.in2_sub2_elem * common.allPoints);
  //printf("%d\n", common.in2_sub2_elem * common.allPoints);
  std::vector<fp> h_d_in2_sqr_sub2((common.in2_sqr_sub2_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in2_sqr_sub2(h_d_in2_sqr_sub2.data(),common.in2_sqr_sub2_mem * common.allPoints);

  //====================================================================================================100
  //  FINAL  (LOCAL)
  //====================================================================================================100

  // common
  common.in_sqr_rows = common.in_rows;
  common.in_sqr_cols = common.in_cols;
  common.in_sqr_elem = common.in_elem;
  common.in_sqr_mem = common.in_mem;

  // unique
  //buffer<fp,1> d_in_sqr(common.in_elem * common.allPoints);
  //printf("%d\n", common.in_elem * common.allPoints);
  std::vector<fp> h_d_in_sqr((common.in_sqr_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in_sqr(h_d_in_sqr.data(),common.in_sqr_mem * common.allPoints);

  //====================================================================================================100
  //  TEMPLATE MASK CREATE  (LOCAL)
  //====================================================================================================100

  // common
  common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
  common.tMask_cols = common.tMask_rows;
  common.tMask_elem = common.tMask_rows * common.tMask_cols;
  common.tMask_mem = sizeof(fp) * common.tMask_elem;

  // unique
  //buffer<fp,1> d_tMask(common.tMask_elem * common.allPoints);
  //printf("%d\n", common.tMask_elem * common.allPoints);
  std::vector<fp> h_d_tMask((common.tMask_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_tMask(h_d_tMask.data(),common.tMask_mem * common.allPoints);

  //====================================================================================================100
  //  POINT MASK INITIALIZE  (LOCAL)
  //====================================================================================================100

  // common
  common.mask_rows = common.maxMove;
  common.mask_cols = common.mask_rows;
  common.mask_elem = common.mask_rows * common.mask_cols;
  common.mask_mem = sizeof(fp) * common.mask_elem;

  //====================================================================================================100
  //  MASK CONVOLUTION  (LOCAL)
  //====================================================================================================100

  // common
  common.mask_conv_rows = common.tMask_rows;                        // number of rows in I
  common.mask_conv_cols = common.tMask_cols;                        // number of columns in I
  common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;                        // number of elements
  common.mask_conv_mem = sizeof(fp) * common.mask_conv_elem;
  common.mask_conv_ioffset = (common.mask_rows-1)/2;
  if((common.mask_rows-1) % 2 > 0.5){
    common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
  }
  common.mask_conv_joffset = (common.mask_cols-1)/2;
  if((common.mask_cols-1) % 2 > 0.5){
    common.mask_conv_joffset = common.mask_conv_joffset + 1;
  }


  //printf("common.endPoints=%d\n", common.endoPoints); // 20
  //printf("common.epiPoints=%d\n", common.epiPoints); // 31
  //printf("common.in_elem=%d\n", common.in_elem);
  //printf("common.endo_mem=%d\n", common.endo_mem); // 80
  //printf("common.epi_mem=%d\n", common.epi_mem); // 124
  //
  //buffer<params_common,1> d_common(&common, 1, props); // range is 1 ?
  //buffer<int,1> d_endoRow(endoRow, common.endoPoints, props);
  //d_endoRow.set_final_data(nullptr);
  //buffer<int,1> d_endoCol(endoCol, common.endoPoints, props);
  //d_endoCol.set_final_data(nullptr);
  //buffer<int,1> d_tEndoRowLoc(tEndoRowLoc, common.endoPoints * common.no_frames, props);
  //buffer<int,1> d_tEndoColLoc(tEndoColLoc, common.endoPoints * common.no_frames, props);
  //buffer<int,1> d_epiRow(epiRow, common.epiPoints, props);
  //d_epiRow.set_final_data(nullptr);
  //buffer<int,1> d_epiCol(epiCol, common.epiPoints, props);
  //d_epiCol.set_final_data(nullptr);
  //buffer<int,1> d_tEpiRowLoc(tEpiRowLoc, common.epiPoints * common.no_frames, props);
  //buffer<int,1> d_tEpiColLoc(tEpiColLoc, common.epiPoints * common.no_frames, props);

  iris::DMem d_endoRow(endoRow,common.endo_mem);

  iris::DMem d_endoCol(endoCol,common.endo_mem);

  iris::DMem d_tEndoRowLoc(tEndoRowLoc,common.endo_mem*common.no_frames);
  iris::DMem d_tEndoColLoc(tEndoColLoc,common.endo_mem*common.no_frames);

  iris::DMem d_epiRow(epiRow,common.epi_mem);
  iris::DMem d_epiCol(epiCol,common.epi_mem);

  iris::DMem d_tEpiRowLoc(tEpiRowLoc,common.epi_mem*common.no_frames);
  iris::DMem d_tEpiColLoc(tEpiColLoc,common.epi_mem*common.no_frames);

  //buffer<fp,1> d_mask_conv(common.mask_conv_elem * common.allPoints);
  //d_mask_conv.set_final_data(nullptr);
  std::vector<fp> h_d_mask_conv((common.mask_conv_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_mask_conv(h_d_mask_conv.data(),common.mask_conv_mem * common.allPoints);

  //printf("%d\n", common.mask_conv_elem * common.allPoints);
  //buffer<fp,1> d_in_mod_temp(common.in_elem * common.allPoints);
  //d_in_mod_temp.set_final_data(nullptr);
  std::vector<fp> h_d_in_mod_temp((common.in_mem * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in_mod_temp(h_d_in_mod_temp.data(),common.in_mem * common.allPoints);

  //printf("%d\n", common.in_elem * common.allPoints);
  //buffer<fp,1> d_in_partial_sum(common.in_cols * common.allPoints);
  //d_in_partial_sum.set_final_data(nullptr);

  std::vector<fp> h_d_in_partial_sum((sizeof(fp)*common.in_cols * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in_partial_sum(h_d_in_partial_sum.data(),sizeof(fp)*common.in_cols * common.allPoints);

  //printf("%d\n", common.in_cols * common.allPoints);
  //buffer<fp,1> d_in_sqr_partial_sum(common.in_sqr_rows * common.allPoints);
  //d_in_sqr_partial_sum.set_final_data(nullptr);

  std::vector<fp> h_d_in_sqr_partial_sum((sizeof(fp)*common.in_sqr_rows * common.allPoints)/sizeof(fp),0);
  iris::DMem d_in_sqr_partial_sum(h_d_in_sqr_partial_sum.data(),sizeof(fp)*common.in_sqr_rows * common.allPoints);


  //printf("%d\n", common.in_sqr_rows * common.allPoints);
  //buffer<fp,1> d_par_max_val(common.mask_conv_rows * common.allPoints);
  //d_par_max_val.set_final_data(nullptr);

  std::vector<fp> h_d_par_max_val((sizeof(fp)*common.mask_conv_rows * common.allPoints)/sizeof(fp),0);
  iris::DMem d_par_max_val(h_d_par_max_val.data(),sizeof(fp)*common.mask_conv_rows * common.allPoints);

  //printf("%d\n", common.mask_conv_rows * common.allPoints);
  //buffer<int,1> d_par_max_coo( common.mask_conv_rows * common.allPoints);
  //d_par_max_coo.set_final_data(nullptr);

  std::vector<fp> h_d_par_max_coo((sizeof(fp)*common.mask_conv_rows * common.allPoints)/sizeof(fp),0);
  iris::DMem d_par_max_coo(h_d_par_max_coo.data(),sizeof(fp)*common.mask_conv_rows * common.allPoints);

  //buffer<fp,1> d_in_final_sum(common.allPoints);
  //d_in_final_sum.set_final_data(nullptr);

  std::vector<fp> h_d_in_final_sum((sizeof(fp)*common.allPoints)/sizeof(fp),0);
  iris::DMem d_in_final_sum(h_d_in_final_sum.data(),sizeof(fp)*common.allPoints);

  //buffer<fp,1> d_in_sqr_final_sum(common.allPoints);
  //d_in_sqr_final_sum.set_final_data(nullptr);
  std::vector<fp> h_d_in_sqr_final_sum((sizeof(fp)*common.allPoints)/sizeof(fp),0);
  iris::DMem d_in_sqr_final_sum(h_d_in_sqr_final_sum.data(),sizeof(fp)*common.allPoints);

  //buffer<fp,1> d_denomT(common.allPoints);
  //d_denomT.set_final_data(nullptr);

  std::vector<fp> h_d_denomT((sizeof(fp)*common.allPoints)/sizeof(fp),0);
  iris::DMem d_denomT(h_d_denomT.data(),sizeof(fp)*common.allPoints);

#ifdef TEST_CHECKSUM
  //buffer<fp,1> d_checksum(CHECK);
  //d_checksum.set_final_data(nullptr);
  //printf("%d\n", CHECK);
  fp* checksum = (fp*) malloc (sizeof(fp)*CHECK);
  std::fill(checksum,checksum+CHECK,fp(0));
  iris::DMem d_checksum(checksum,sizeof(fp)*CHECK);
#endif

  //====================================================================================================100
  //  EXECUTION PARAMETERS
  //====================================================================================================100

  // All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
  int device=hip_device();
  size_t gws=size_t(common.allPoints)*NUMBER_THREADS,lws=NUMBER_THREADS;


  printf("frame progress: ");
  fflush(NULL);

  //====================================================================================================100
  //  LAUNCH
  //====================================================================================================100

  // variables
  fp* frame;
  int frame_no;

  //buffer<fp,1> d_frame(common.frame_elem);
  std::vector<fp> h_d_frame((sizeof(fp)*common.frame_elem)/sizeof(fp),0);
  iris::DMem d_frame(h_d_frame.data(),sizeof(fp)*common.frame_elem);

  uint64_t kernel_time = 0;

  for(frame_no=0; frame_no<common.frames_processed; frame_no++) {

    //==================================================50
    //  get and write current frame to GPU buffer
    //==================================================50

    // Extract a cropped version of the first frame from the video file
    frame = get_frame(  frames,                // pointer to video file
        frame_no,              // number of frame that needs to be returned
        0,                  // cropped?
        0,                  // scaled?
        1);                  // converted

    // copy frame to GPU memory
    std::copy(frame,frame+common.frame_elem,h_d_frame.begin());
    check(d_frame.update(h_d_frame.data()));

    //==================================================50
    //  launch kernel
    //==================================================50
    auto start_time = std::chrono::steady_clock::now();

    void* args[]={&frame_no,&common,&d_frame,&d_endoRow,&d_endoCol,&d_tEndoRowLoc,&d_tEndoColLoc,&d_epiRow,&d_epiCol,&d_tEpiRowLoc,&d_tEpiColLoc,&d_endoT,&d_epiT,&d_in2,&d_conv,&d_in2_pad_cumv,&d_in2_pad_cumv_sel,&d_in2_sub_cumh,&d_in2_sub_cumh_sel,&d_in2_sub2,&d_in2_sqr,&d_in2_sqr_sub2,&d_in_sqr,&d_tMask,&d_mask_conv,&d_in_mod_temp,&d_in_partial_sum,&d_in_sqr_partial_sum,&d_par_max_val,&d_par_max_coo,&d_in_final_sum,&d_in_sqr_final_sum,&d_denomT
#ifdef TEST_CHECKSUM
      ,&d_checksum
#endif
    };
    int info[]={sizeof(frame_no),sizeof(common),iris_r,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw
#ifdef TEST_CHECKSUM
      ,iris_rw
#endif
    };
    iris::Task task;check(task.kernel("hw",1,nullptr,&gws,&lws,sizeof(args)/sizeof(args[0]),args,info));check(task.submit(device,nullptr,true));
    auto end_time = std::chrono::steady_clock::now();
    kernel_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    // free frame after each loop iteration, since AVI library allocates memory for every frame fetched
    free(frame);

    //==================================================50
    //  print frame progress
    //==================================================50

    // print frame progress
    printf("%d ", frame_no);
    fflush(NULL);

    //==================================================50
    //  DISPLAY CHECKSUM (TESTING)
    //==================================================50

#ifdef TEST_CHECKSUM
    iris::Task csflush;check(csflush.flush_out(d_checksum));check(csflush.submit(device,nullptr,true));
    printf("CHECKSUM:\n");
    for(int i=0; i<CHECK; i++){
      printf("i=%d checksum=%f\n", i, checksum[i]);
    }
    printf("\n\n");
#endif

  }



  //====================================================================================================100
  //  PRINT FRAME PROGRESS END
  iris::Task flush;
  check(flush.flush_out(d_tEndoRowLoc));
  check(flush.flush_out(d_tEndoColLoc));
  check(flush.flush_out(d_tEpiRowLoc));
  check(flush.flush_out(d_tEpiColLoc));
  check(flush.submit(device,nullptr,true));
  //====================================================================================================100
#ifdef TEST_CHECKSUM
  free(checksum);
#endif

  printf("\n");
  fflush(NULL);
  return kernel_time;
}
