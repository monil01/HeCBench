#include "bm3d.hpp"
extern "C" void run_block_matching(
    BMBuffer &image, //Original image
    BMBuffer &stacks,         //For each reference patch contains addresses of similar patches (patch is adressed by top left corner)
    BMBuffer &num_patches_in_stack,    //For each reference patch contains number of similar patches
    uint2 image_dim,      //Image dimensions
    uint2 stacks_dim,      //size of area where reference patches could be located
    Params params,      //Denoising parameters
    uint2 start_point,    //Address of the top-left reference patch of a batch
    dim3 num_threads,
    dim3 num_blocks,
    uint shared_memory_size
    )
{
    void *args[] = { &image.mem, &stacks.mem, &num_patches_in_stack.mem, &image_dim, &stacks_dim, &params, &start_point };
  int info[] = { iris_r, iris_rw, iris_rw, sizeof(image_dim), sizeof(stacks_dim), sizeof(params), sizeof(start_point) };
  size_t gws[] = { size_t(num_blocks.x)*num_threads.x,size_t(num_blocks.y)*num_threads.y,size_t(num_blocks.z)*num_threads.z };
  size_t lws[] = { num_threads.x,num_threads.y,num_threads.z };
  iris::Task task;
  iris_check(task.kernel("block_matching",3,nullptr,gws,lws,7,args,info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}

extern "C" void run_get_block(
  uint2 start_point,
  BMBuffer &image,
  BMBuffer &stacks,
  BMBuffer &num_patches_in_stack,
  BMBuffer &patch_stack,
  uint2 image_dim,
  uint2 stacks_dim,
  Params params,
  dim3 num_threads,
  dim3 num_blocks)
{
    void *args[] = { &start_point, &image.mem, &stacks.mem, &num_patches_in_stack.mem, &patch_stack.mem, &image_dim, &stacks_dim, &params };
  int info[] = { sizeof(start_point), iris_r, iris_rw, iris_rw, iris_rw, sizeof(image_dim), sizeof(stacks_dim), sizeof(params) };
  size_t gws[] = { size_t(num_blocks.x)*num_threads.x,size_t(num_blocks.y)*num_threads.y,size_t(num_blocks.z)*num_threads.z };
  size_t lws[] = { num_threads.x,num_threads.y,num_threads.z };
  iris::Task task;
  iris_check(task.kernel("get_block",3,nullptr,gws,lws,8,args,info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}

extern "C" void run_hard_treshold_block(
  uint2 start_point,
  BMBuffer &patch_stack,
  BMBuffer &w_P,
  BMBuffer &num_patches_in_stack,
  uint2 stacks_dim,
  Params params,
  uint sigma,
  dim3 num_threads,
  dim3 num_blocks,
  uint shared_memory_size)
{
    void *args[] = { &start_point, &patch_stack.mem, &w_P.mem, &num_patches_in_stack.mem, &stacks_dim, &params, &sigma };
  int info[] = { sizeof(start_point), iris_rw, iris_rw, iris_rw, sizeof(stacks_dim), sizeof(params), sizeof(sigma) };
  size_t gws[] = { size_t(num_blocks.x)*num_threads.x,size_t(num_blocks.y)*num_threads.y,size_t(num_blocks.z)*num_threads.z };
  size_t lws[] = { num_threads.x,num_threads.y,num_threads.z };
  iris::Task task;
  iris_check(task.kernel("hard_treshold_block",3,nullptr,gws,lws,7,args,info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}

extern "C" void run_aggregate_block(
  uint2 start_point,
  BMBuffer &patch_stack,
  BMBuffer &w_P,
  BMBuffer &stacks,
  BMBuffer &kaiser_window,
  BMBuffer &numerator,
  BMBuffer &denominator,
  BMBuffer &num_patches_in_stack,
  uint2 image_dim,
  uint2 stacks_dim,
  Params params,
  dim3 num_threads,
  dim3 num_blocks)
{
    void *args[] = { &start_point, &patch_stack.mem, &w_P.mem, &stacks.mem, &kaiser_window.mem, &numerator.mem, &denominator.mem, &num_patches_in_stack.mem, &image_dim, &stacks_dim, &params };
  int info[] = { sizeof(start_point), iris_rw, iris_rw, iris_rw, iris_r, iris_rw, iris_rw, iris_rw, sizeof(image_dim), sizeof(stacks_dim), sizeof(params) };
  size_t gws[] = { size_t(num_blocks.x)*num_threads.x,size_t(num_blocks.y)*num_threads.y,size_t(num_blocks.z)*num_threads.z };
  size_t lws[] = { num_threads.x,num_threads.y,num_threads.z };
  iris::Task task;
  iris_check(task.kernel("aggregate_block",3,nullptr,gws,lws,11,args,info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}

extern "C" void run_aggregate_final(
  BMBuffer &numerator,
  BMBuffer &denominator,
  uint2 image_dim,
  BMBuffer &denoised_image,
  dim3 num_threads,
  dim3 num_blocks
)
{
    void *args[] = { &numerator.mem, &denominator.mem, &image_dim, &denoised_image.mem };
  int info[] = { iris_rw, iris_rw, sizeof(image_dim), iris_rw };
  size_t gws[] = { size_t(num_blocks.x)*num_threads.x,size_t(num_blocks.y)*num_threads.y,size_t(num_blocks.z)*num_threads.z };
  size_t lws[] = { num_threads.x,num_threads.y,num_threads.z };
  iris::Task task;
  iris_check(task.kernel("aggregate_final",3,nullptr,gws,lws,4,args,info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}


extern "C" void run_DCT2D8x8(
  BMBuffer &transformed_stacks,
  BMBuffer &gathered_stacks,
  uint size,
  dim3 num_threads,
  dim3 num_blocks)
{
    void *args[] = { &transformed_stacks.mem, &gathered_stacks.mem, &size };
  int info[] = { iris_rw, iris_r, sizeof(size) };
  size_t gws[] = { size_t(num_blocks.x)*num_threads.x,size_t(num_blocks.y)*num_threads.y,size_t(num_blocks.z)*num_threads.z };
  size_t lws[] = { num_threads.x,num_threads.y,num_threads.z };
  iris::Task task;
  iris_check(task.kernel("DCT2D8x8",3,nullptr,gws,lws,3,args,info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}

extern "C" void run_IDCT2D8x8(
  BMBuffer &gathered_stacks,
  BMBuffer &transformed_stacks,
  uint size,
  dim3 num_threads,
  dim3 num_blocks)
{
    void *args[] = { &gathered_stacks.mem, &transformed_stacks.mem, &size };
  int info[] = { iris_rw, iris_r, sizeof(size) };
  size_t gws[] = { size_t(num_blocks.x)*num_threads.x,size_t(num_blocks.y)*num_threads.y,size_t(num_blocks.z)*num_threads.z };
  size_t lws[] = { num_threads.x,num_threads.y,num_threads.z };
  iris::Task task;
  iris_check(task.kernel("IDCT2D8x8",3,nullptr,gws,lws,3,args,info));
  iris_check(task.submit(iris_gpu,nullptr,true));
}
