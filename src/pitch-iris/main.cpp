#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <stdexcept>

static constexpr int kBlock2DX = 16;
static constexpr int kBlock2DY = 16;
static constexpr int kBlock3DX = 16;
static constexpr int kBlock3DY = 8;
static constexpr int kBlock3DZ = 4;

static void diagnostic_dump(const char* name,iris::DMem& mem,const std::vector<float>& data,int width,int height,int depth,size_t pitch) {
 const char* prefix=getenv("HECBENCH_DUMP"); if(!prefix) return;
 iris::Task flush;flush.flush_out(mem);
 if(flush.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) throw std::runtime_error("flush failed");
 std::vector<float> compact(size_t(width)*height*depth);
 for(int z=0;z<depth;z++) for(int y=0;y<height;y++)
  for(int x=0;x<width;x++) compact[(size_t(z)*height+y)*width+x]=data[(size_t(z)*height+y)*(pitch/sizeof(float))+x];
 std::string path=std::string(prefix)+"."+name+"."+std::to_string(width)+"x"+std::to_string(height)+"x"+std::to_string(depth)+".raw";
 FILE* f=fopen(path.c_str(),"wb");if(!f || fwrite(compact.data(),sizeof(float),compact.size(),f)!=compact.size()) throw std::runtime_error("dump failed");fclose(f);
}

static void launch_repeated(
    const char* kernel_name,
    size_t gx,
    size_t gy,
    size_t gz,
    size_t lx,
    size_t ly,
    size_t lz,
    int repeat,
    void** params,
    int* pinfo,
    int nparams)
{
  const size_t gws[3] = { gx, gy, gz };
  const size_t lws[3] = { lx, ly, lz };
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    task.kernel(kernel_name, 3, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                nparams, params, pinfo);
    if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) throw std::runtime_error(kernel_name);
  }
}

static void malloc2D(int repeat, int width, int height)
{
  std::printf("Dimension: (%d %d)\n", width, height);

  const size_t gx = static_cast<size_t>((width + 15) / 16) * kBlock2DX;
  const size_t gy = static_cast<size_t>((height + 15) / 16) * kBlock2DY;
  const size_t pitch = ((static_cast<size_t>(width)*sizeof(float)+127)/128)*128;
  const size_t count = static_cast<size_t>(width) * height;

  std::vector<float> pitched((pitch/sizeof(float))*height,0.0f);
  iris::DMem mem_pitched(pitched.data(),pitched.size()*sizeof(float));
  void* pitched_params[4] = { &mem_pitched, const_cast<size_t*>(&pitch), &width, &height };
  int pitched_pinfo[4] = { iris_rw, static_cast<int>(sizeof(pitch)), static_cast<int>(sizeof(width)),
                           static_cast<int>(sizeof(height)) };

  launch_repeated("parallelPitched2DAccess", gx, gy, 1, kBlock2DX, kBlock2DY, 1, 1,
                  pitched_params, pitched_pinfo, 4);

  auto start = std::chrono::steady_clock::now();
  launch_repeated("parallelPitched2DAccess", gx, gy, 1, kBlock2DX, kBlock2DY, 1, repeat,
                  pitched_params, pitched_pinfo, 4);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  diagnostic_dump("pitched2D",mem_pitched,pitched,width,height,1,pitch);
  std::vector<float> simple(count, 0.0f);
  iris::DMem mem_simple(simple.data(), count * sizeof(float));
  void* simple_params[3] = { &mem_simple, &width, &height };
  int simple_pinfo[3] = { iris_rw, static_cast<int>(sizeof(width)), static_cast<int>(sizeof(height)) };

  launch_repeated("parallelSimple2DAccess", gx, gy, 1, kBlock2DX, kBlock2DY, 1, 1,
                  simple_params, simple_pinfo, 3);

  start = std::chrono::steady_clock::now();
  launch_repeated("parallelSimple2DAccess", gx, gy, 1, kBlock2DX, kBlock2DY, 1, repeat,
                  simple_params, simple_pinfo, 3);
  end = std::chrono::steady_clock::now();
  auto time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  diagnostic_dump("simple2D",mem_simple,simple,width,height,1,size_t(width)*sizeof(float));
  std::printf("Average execution time (pitched vs simple): %f %f (us)\n",
              (time * 1e-3f) / repeat, (time2 * 1e-3f) / repeat);
}

static void malloc3D(int repeat, int width, int height, int depth)
{
  std::printf("Dimension: (%d %d %d)\n", width, height, depth);

  const size_t gx = static_cast<size_t>((width + 15) / 16) * kBlock3DX;
  const size_t gy = static_cast<size_t>((height + 7) / 8) * kBlock3DY;
  const size_t gz = static_cast<size_t>((depth + 3) / 4) * kBlock3DZ;
  const size_t pitch = ((static_cast<size_t>(width)*sizeof(float)+127)/128)*128;
  const size_t count = static_cast<size_t>(width) * height * depth;

  std::vector<float> pitched((pitch/sizeof(float))*height*depth,0.0f);
  iris::DMem mem_pitched(pitched.data(),pitched.size()*sizeof(float));
  void* pitched_params[5] = { &mem_pitched, const_cast<size_t*>(&pitch), &width, &height, &depth };
  int pitched_pinfo[5] = { iris_rw, static_cast<int>(sizeof(pitch)), static_cast<int>(sizeof(width)),
                           static_cast<int>(sizeof(height)), static_cast<int>(sizeof(depth)) };

  launch_repeated("parallelPitched3DAccess", gx, gy, gz, kBlock3DX, kBlock3DY, kBlock3DZ, 1,
                  pitched_params, pitched_pinfo, 5);

  auto start = std::chrono::steady_clock::now();
  launch_repeated("parallelPitched3DAccess", gx, gy, gz, kBlock3DX, kBlock3DY, kBlock3DZ, repeat,
                  pitched_params, pitched_pinfo, 5);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  diagnostic_dump("pitched3D",mem_pitched,pitched,width,height,depth,pitch);
  std::vector<float> simple(count, 0.0f);
  iris::DMem mem_simple(simple.data(), count * sizeof(float));
  void* simple_params[4] = { &mem_simple, &width, &height, &depth };
  int simple_pinfo[4] = { iris_rw, static_cast<int>(sizeof(width)), static_cast<int>(sizeof(height)),
                          static_cast<int>(sizeof(depth)) };

  launch_repeated("parallelSimple3DAccess", gx, gy, gz, kBlock3DX, kBlock3DY, kBlock3DZ, 1,
                  simple_params, simple_pinfo, 4);

  start = std::chrono::steady_clock::now();
  launch_repeated("parallelSimple3DAccess", gx, gy, gz, kBlock3DX, kBlock3DY, kBlock3DZ, repeat,
                  simple_params, simple_pinfo, 4);
  end = std::chrono::steady_clock::now();
  auto time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  diagnostic_dump("simple3D",mem_simple,simple,width,height,depth,size_t(width)*sizeof(float));
  std::printf("Average execution time (pitched vs simple): %f %f (us)\n",
              (time * 1e-3f) / repeat, (time2 * 1e-3f) / repeat);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const char* shape_env=getenv("HECBENCH_PITCH_SHAPES");
  const int shape_count=shape_env?std::atoi(shape_env):11;
  if(shape_count<1 || shape_count>11 || repeat<1) return 1;
  const int w[] = {227, 256, 720, 768, 854, 1280, 1440, 1920, 2048, 3840, 4096};
  const int h[] = {227, 256, 480, 576, 480, 720, 1080, 1080, 1080, 2160, 2160};
  const int d[] = {1, 3};

  for (int i = 0; i < shape_count; i++) {
    malloc2D(repeat, w[i], h[i]);
  }

  for (int i = 0; i < shape_count; i++) {
    for (int j = 0; j < 2; j++) {
      malloc3D(repeat, w[i], h[i], d[j]);
    }
  }

  platform.finalize();
  return 0;
}
