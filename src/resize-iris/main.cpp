#include <iris/iris.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <string>
#include <cstring>
#include <stdexcept>
template<class T> void diagnostic_dump(const T* output,size_t count,bool bilinear) {
 const char* prefix=getenv("HECBENCH_DUMP");
 if(!prefix) return;
 std::string path=std::string(prefix)+".u"+std::to_string(sizeof(T)*8)+(bilinear?".bilinear.raw":".nearest.raw");
 FILE* f=fopen(path.c_str(),"wb");
 if(!f || fwrite(output,sizeof(T),count,f)!=count) throw std::runtime_error("output dump failed");
 fclose(f);
}
template <class T>
void resize_image (
  const int in_width,
  const int in_height,
  const int out_width,
  const int out_height,
  const int num_channels,
  const int repeat,
  const bool bilinear = false)
{
  size_t in_image_size = (size_t)in_height * in_width;
  size_t in_size = num_channels * in_image_size;
  size_t in_size_bytes = sizeof(T) * in_size;

  size_t out_image_size = (size_t)out_height * out_width;
  size_t out_size = num_channels * out_image_size;
  size_t out_size_bytes = sizeof(T) * out_size;

  T* in_images_h = (T*) malloc (in_size_bytes);
  T* out_images_h = (T*) malloc (out_size_bytes);

  for(size_t i = 0; i < in_size; i++) in_images_h[i] = static_cast<T>((i+1) % 13);

  memset(out_images_h,0,out_size_bytes);
  iris::DMem in_images_d(in_images_h,in_size_bytes),out_images_d(out_images_h,out_size_bytes);

  const float fx = in_width / out_width;
  const float fy = in_height / out_height;


  auto start = std::chrono::steady_clock::now();

  size_t global[]={29184*256},local[]={256};
  bool round=true,half_pixel_centers=true;
  std::string name=std::string(bilinear?"resize_bilinear_u":"resize_u")+std::to_string(sizeof(T)*8);
  for(int it=0;it<repeat;it++) {
   iris::Task task;
   void* args[]={&out_images_d,&out_size,(void*)&out_height,(void*)&out_width,&in_images_d,(void*)&in_height,(void*)&in_width,(void*)&fx,(void*)&fy,&round,&half_pixel_centers};
   int info[]={iris_rw,sizeof(out_size),sizeof(out_height),sizeof(out_width),iris_r,sizeof(in_height),sizeof(in_width),sizeof(fy),sizeof(fx),sizeof(round),sizeof(half_pixel_centers)};
   if(bilinear) args[9]=&half_pixel_centers;
   task.kernel(name.c_str(),1,nullptr,global,local,bilinear?10:11,args,info);
   if(it==repeat-1) task.flush_out(out_images_d);
   if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) throw std::runtime_error(name);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %lf (us)    Perf: %lf (GB/s)\n",
         time * 1e-3 / repeat, (in_size_bytes + out_size_bytes) * repeat * 1.0 / time);

  diagnostic_dump(out_images_h,out_size,bilinear);




  free(in_images_h);
  free(out_images_h);
}

int main(int argc, char* argv[]) {
  if (argc != 7) {
    printf("Usage: %s <input image width> <input image height>\n", argv[0]);
    printf("          <output image width> <output image height>\n");
    printf("          <image channels> <repeat>\n");
    return 1;
  }

  const int in_width = atoi(argv[1]);
  const int in_height = atoi(argv[2]);
  const int out_width = atoi(argv[3]);
  const int out_height = atoi(argv[4]);
  const int num_channels = atoi(argv[5]);
  const int repeat = atoi(argv[6]);

  iris::Platform platform;platform.init(&argc,&argv,1);
  printf("Resize %d images from (%d x %d) to (%d x %d)\n",
          num_channels, in_width, in_height, out_width, out_height);

  printf("\nThe size of each pixel is 1 byte\n");
  resize_image<unsigned char>(in_width, in_height, out_width, out_height, num_channels, repeat);
  printf("\nBilinear resizing\n");
  resize_image<unsigned char>(in_width, in_height, out_width, out_height, num_channels, repeat, true);

  printf("\nThe size of each pixel is 2 bytes\n");
  resize_image<unsigned short>(in_width, in_height, out_width, out_height, num_channels, repeat);
  printf("\nBilinear resizing\n");
  resize_image<unsigned short>(in_width, in_height, out_width, out_height, num_channels, repeat, true);

  printf("\nThe size of each pixel is 4 bytes\n");
  resize_image<unsigned int>(in_width, in_height, out_width, out_height, num_channels, repeat);
  printf("\nBilinear resizing\n");
  resize_image<unsigned int>(in_width, in_height, out_width, out_height, num_channels, repeat, true);

  platform.finalize();
  return 0;
}
