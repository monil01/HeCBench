#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <stdexcept>
#define BLOCK_SIZE 256
#define integrateXcoord 1
#define integrateYcoord 0
struct dim3 {size_t x,y,z; dim3(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c){}};
static int gpu_device = -1;
struct Param {void* ptr; int info;};
static Param param(iris::DMem& mem) {return {&mem, iris_rw};}
template<class T> static Param param(const T& value) {return {const_cast<T*>(&value), sizeof(T)};}
template<class... Args> static void launch(const char* name, dim3 grid, dim3 block, Args&&... args) {
  Param p[] = {param(args)...};
  void* values[sizeof...(args)]; int info[sizeof...(args)];
  for (size_t i=0;i<sizeof...(args);++i) {values[i]=p[i].ptr; info[i]=p[i].info;}
  size_t global[] = {grid.x*block.x,grid.y*block.y,grid.z*block.z};
  size_t local[] = {block.x,block.y,block.z};
  iris::Task task;
  if(task.kernel(name,3,nullptr,global,local,sizeof...(args),values,info) ||
     task.submit(gpu_device,nullptr,1)) throw std::runtime_error("IRIS kernel task failed");
}
static void flush(iris::DMem& mem) {
  iris::Task task; task.flush_out(mem);
  if(task.submit(gpu_device,nullptr,1)) throw std::runtime_error("IRIS output flush failed");
}
static int validation_size(const char* name, int fallback) {
  const char* value = getenv(name);
  return value ? atoi(value) : fallback;
}
// Branchless distance-driven backprojection 
void backprojectionDDb(
          double* const h_pVolume,
    const double* const h_pProj,
    const double* const h_pTubeAngle,
    const double* const h_pDetAngle,
    const int idXProj,
    const int nProj,
    const int nPixX,
    const int nPixY,
    const int nSlices,
    const int nDetX,
    const int nDetY,
    const double dx,
    const double dy,
    const double dz,
    const double du,
    const double dv,
    const double DSD,
    const double DDR,
    const double DAG)
{
  // Number of mapped detectors
  const int nDetXMap = nDetX + 1;
  const int nDetYMap = nDetY + 1;

  // Number of mapped pixels
  const int nPixXMap = nPixX + 1;
  const int nPixYMap = nPixY + 1;


  std::vector<double> d_pProj_host((nDetXMap*nDetYMap*nProj * sizeof(double)) / sizeof(double));
  iris::DMem d_pProj(d_pProj_host.data(), nDetXMap*nDetYMap*nProj * sizeof(double)); 
  std::vector<double> d_sliceI_host((nPixXMap*nPixYMap * sizeof(double)) / sizeof(double));
  iris::DMem d_sliceI(d_sliceI_host.data(), nPixXMap*nPixYMap * sizeof(double));
  std::vector<double> d_pVolume_host((nPixX*nPixY*nSlices * sizeof(double)) / sizeof(double));
  iris::DMem d_pVolume(d_pVolume_host.data(), nPixX*nPixY*nSlices * sizeof(double));

  // device memory for projections coordinates

  std::vector<double> d_pDetX_host((nDetXMap * sizeof(double)) / sizeof(double));
  iris::DMem d_pDetX(d_pDetX_host.data(), nDetXMap * sizeof(double));
  std::vector<double> d_pDetY_host((nDetYMap * sizeof(double)) / sizeof(double));
  iris::DMem d_pDetY(d_pDetY_host.data(), nDetYMap * sizeof(double));
  std::vector<double> d_pDetZ_host((nDetYMap * sizeof(double)) / sizeof(double));
  iris::DMem d_pDetZ(d_pDetZ_host.data(), nDetYMap * sizeof(double));
  std::vector<double> d_pObjX_host((nPixXMap * sizeof(double)) / sizeof(double));
  iris::DMem d_pObjX(d_pObjX_host.data(), nPixXMap * sizeof(double));
  std::vector<double> d_pObjY_host((nPixYMap * sizeof(double)) / sizeof(double));
  iris::DMem d_pObjY(d_pObjY_host.data(), nPixYMap * sizeof(double));
  std::vector<double> d_pObjZ_host((nSlices * sizeof(double)) / sizeof(double));
  iris::DMem d_pObjZ(d_pObjZ_host.data(), nSlices * sizeof(double));

  // device memory for mapped coordinates

  std::vector<double> d_pDetmY_host((nDetYMap * sizeof(double)) / sizeof(double));
  iris::DMem d_pDetmY(d_pDetmY_host.data(), nDetYMap * sizeof(double));
  std::vector<double> d_pDetmX_host((nDetYMap * nDetXMap * sizeof(double)) / sizeof(double));
  iris::DMem d_pDetmX(d_pDetmX_host.data(), nDetYMap * nDetXMap * sizeof(double));

  // device memory for rotated detector coords

  std::vector<double> d_pRdetY_host((nDetYMap * sizeof(double)) / sizeof(double));
  iris::DMem d_pRdetY(d_pRdetY_host.data(), nDetYMap * sizeof(double));
  std::vector<double> d_pRdetZ_host((nDetYMap * sizeof(double)) / sizeof(double));
  iris::DMem d_pRdetZ(d_pRdetZ_host.data(), nDetYMap * sizeof(double));

  auto start = std::chrono::steady_clock::now();

  // Will reuse grid configurations
  dim3 threadsPerBlock (1,1,1);
  dim3 blockSize (1,1,1);

  const int maxThreadsPerBlock = BLOCK_SIZE;

  // Pad on the host and publish through DMem, preserving the mapped layout.
  for (int np = 0; np < nProj; ++np)
    for (int c = 0; c < nDetX; ++c)
      std::copy(h_pProj + c*nDetY + nDetX*nDetY*np,
                h_pProj + c*nDetY + nDetX*nDetY*np + nDetY,
                d_pProj_host.data() + (c+1)*nDetYMap + 1 + nDetXMap*nDetYMap*np);
  d_pProj.update(d_pProj_host.data());
  // Generate detector and object boudaries

  threadsPerBlock.x = maxThreadsPerBlock;

  blockSize.x = (nDetX / maxThreadsPerBlock) + 1;

  launch("map_boudaries_kernel", blockSize, threadsPerBlock, d_pDetX, nDetXMap, (double)nDetX, -du, 0.0);

  blockSize.x = (nDetY / maxThreadsPerBlock) + 1;

  launch("map_boudaries_kernel", blockSize, threadsPerBlock, d_pDetY, nDetYMap, nDetY / 2.0, dv, 0.0);

  blockSize.x = (nPixX / maxThreadsPerBlock) + 1;

  launch("map_boudaries_kernel", blockSize, threadsPerBlock, d_pObjX, nPixXMap, (double)nPixX, -dx, 0.0);

  blockSize.x = (nPixY / maxThreadsPerBlock) + 1;

  launch("map_boudaries_kernel", blockSize, threadsPerBlock, d_pObjY, nPixYMap, nPixY / 2.0, dy, 0.0);

  blockSize.x = (nSlices / maxThreadsPerBlock) + 1;

  launch("map_boudaries_kernel", blockSize, threadsPerBlock, d_pObjZ, nSlices, 0.0, dz, DAG + (dz / 2.0));

  // Initiate variables value with 0

  // X - ray tube initial position
  double tubeX = 0;
  double tubeY = 0;
  double tubeZ = DSD;

  // Iso - center position
  double isoY = 0;
  double isoZ = DDR;

  // Integration of 2D projection over the whole projections
  // (S.1.Integration. - Liu et al(2017))

  // Naive integration o the X coord
  threadsPerBlock.x = 8;
  threadsPerBlock.y = 4;
  threadsPerBlock.z = 8;

  blockSize.x = (int)ceilf((float)nDetYMap / (threadsPerBlock.x - 1));
  blockSize.y = 1;
  blockSize.z = (int)ceilf((float)nProj / threadsPerBlock.z);

  int Xk = (int)ceilf((float)nDetXMap / (threadsPerBlock.x - 1));
  for (int k = 0; k < Xk; k++) {

    launch("img_integration_kernel", blockSize, threadsPerBlock, d_pProj, nDetXMap, nDetYMap, integrateXcoord, 0, k * 9, nProj);
  }

  // Naive integration o the Y coord
  threadsPerBlock.x = 4;
  threadsPerBlock.y = 8;
  threadsPerBlock.z = 8;

  blockSize.x = 1;
  blockSize.y = (int)ceilf((float)nDetXMap / (threadsPerBlock.y - 1));
  blockSize.z = (int)ceilf((float)nProj / threadsPerBlock.z);

  int Yk = (int)ceilf((float)nDetYMap / (threadsPerBlock.y - 1));
  for (int k = 0; k < Yk; k++) {

    launch("img_integration_kernel", blockSize, threadsPerBlock, d_pProj, nDetXMap, nDetYMap, integrateYcoord, k * 9, 0, nProj);
  }

  const int detmOffset = nDetYMap * (nDetXMap-2);

  int projIni, projEnd, nProj2Run;
  if (idXProj == -1) {
    projIni = 0;
    projEnd = nProj;
    nProj2Run = nProj;
  }
  else {
    projIni = idXProj;
    projEnd = idXProj + 1;
    nProj2Run = 1;
  }

  // For each projection
  for (int p = projIni; p < projEnd; p++) {

    // Get specif tube angle for the projection
    double theta = h_pTubeAngle[p] * M_PI / 180.0;

    // Get specif detector angle for the projection
    double phi = h_pDetAngle[p] * M_PI / 180.0;

    //printf("Tube angle:%f Det angle:%f\n", theta, phi);

    // Tube rotation
    double rtubeY = ((tubeY - isoY)*cos(theta) - (tubeZ - isoZ)*sin(theta)) + isoY;
    double rtubeZ = ((tubeY - isoY)*sin(theta) + (tubeZ - isoZ)*cos(theta)) + isoZ;

    //printf("R tube Y:%f R tube Z:%f\n", rtubeY, rtubeZ);

    // Detector rotation
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blockSize.x = (nDetYMap / maxThreadsPerBlock) + 1;
    blockSize.y = 1;
    blockSize.z = 1;

    launch("rot_detector_kernel", blockSize, threadsPerBlock, d_pRdetY, d_pRdetZ, d_pDetY, d_pDetZ, isoY, isoZ, phi, nDetYMap);

    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    threadsPerBlock.z = 1;

    // For each slice
    for (int nz = 0; nz < nSlices; nz++) {

      // Map detector onto XY plane(Inside proj loop in case detector rotates)

      blockSize.x = (nDetYMap / threadsPerBlock.x) + 1;
      blockSize.y = (nDetXMap / threadsPerBlock.y) + 1;
      blockSize.z = 1;

      launch("mapDet2Slice_kernel", blockSize, threadsPerBlock, d_pDetmX, d_pDetmY, tubeX, rtubeY, rtubeZ, d_pDetX,
          d_pRdetY, d_pRdetZ, d_pObjZ, nDetXMap, nDetYMap, nz);

      //  S.2. Interpolation - Liu et al (2017)

      blockSize.x = (nPixYMap / threadsPerBlock.x) + 1;
      blockSize.y = (nPixXMap / threadsPerBlock.y) + 1;

      launch("bilinear_interpolation_kernel", blockSize, threadsPerBlock, d_sliceI, d_pProj, d_pObjX, d_pObjY, d_pDetmX, d_pDetmY,
          nPixXMap, nPixYMap, nDetXMap, nDetYMap, nDetX, nDetY, p, detmOffset);

      // S.3. Differentiation - Eq. 24 - Liu et al (2017)

      blockSize.x = (nPixY / threadsPerBlock.x) + 1;
      blockSize.y = (nPixX / threadsPerBlock.y) + 1;

      launch("differentiation_kernel", blockSize, threadsPerBlock, d_pVolume, d_sliceI, tubeX, rtubeY, rtubeZ, d_pObjX, d_pObjY, d_pObjZ,
          nPixX, nPixY, nPixXMap, nPixYMap, du, dv, dx, dy, dz, nz);

    } // Loop end slices

  } // Loop end Projections


  // Normalize volume dividing by the number of projections
  threadsPerBlock.x = 8;
  threadsPerBlock.y = 8;
  threadsPerBlock.z = 4;

  blockSize.x = (nPixY / threadsPerBlock.x) + 1;
  blockSize.y = (nPixX / threadsPerBlock.y) + 1;
  blockSize.z = (nSlices / threadsPerBlock.z) + 1;

  launch("division_kernel", blockSize, threadsPerBlock, d_pVolume, nPixX, nPixY, nSlices, nProj2Run);

  flush(d_pVolume);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution %f (s)\n", time * 1e-9f);

  std::copy(d_pVolume_host.begin(), d_pVolume_host.end(), h_pVolume);

}

int main() 
{
                            // image voxel density
  const int nPixX = validation_size("nPixX", 1996);   // number of voxels
  const int nPixY = validation_size("nPixY", 2457);   // number of voxels
  const int nSlices = validation_size("nSlices", 78);  

                            // detector panel pixel density
  const int nDetX = validation_size("nDetX", 1664);   // number of pixels
  const int nDetY = validation_size("nDetY", 2048);   // number of pixels

  const int nProj = validation_size("nProj", 15);     // number of projections
  const int idXProj = -1;   // loop over all projections

  const double dx = 0.112;  // single voxel size (mm)
  const double dy = 0.112;
  const double dz = 1.0;

  const double du = 0.14;   // single detector size (mm)
  const double dv = 0.14;

  const double DSD = 700;   // distance from source to detector (mm)
  const double DDR = 0.0;   // distance from detector to pivot (mm)
  const double DAG = 25.0;  // distance of air gap (mm)

  const size_t pixVol = nPixX * nPixY * nSlices;
  const size_t detVol = nDetX * nDetY * nProj;
  double *h_pVolume = (double*) malloc (pixVol * sizeof(double));
  double *h_pProj = (double*) malloc (detVol * sizeof(double));

  double *h_pTubeAngle = (double*) malloc (nProj * sizeof(double));
  double *h_pDetAngle = (double*) malloc (nProj * sizeof(double));
  
  // tube angles in degrees
  for (int i = 0; i < nProj; i++) 
    h_pTubeAngle[i] = -7.5 + i * 15.0/nProj;

  // detector angles in degrees
  for (int i = 0; i < nProj; i++) 
    h_pDetAngle[i] = -2.1 + i * 4.2/nProj;

  // random values
  srand(123);
  for (size_t i = 0; i < pixVol; i++) 
    h_pVolume[i] = (double)rand() / (double)RAND_MAX;

  for (size_t i = 0; i < detVol; i++) 
    h_pProj[i] = (double)rand() / (double)RAND_MAX;

  iris::Platform platform;
  int argc = 0; char** argv = nullptr;
  if (platform.init(&argc, &argv, 1)) return 2;
  int count = 0; iris_device_count(&count);
  for (int i = 0; i < count; ++i) {
    int backend = 0, type = 0; char name[256] = {};
    iris_device_info(i, iris_backend, &backend, nullptr);
    iris_device_info(i, iris_type, &type, nullptr);
    iris_device_info(i, iris_name, name, nullptr);
    fprintf(stderr, "IRIS device %d backend=%d type=%d name=%s\n", i, backend, type, name);
    if (backend == 3 && type == 256) gpu_device = i;
  }
  if (gpu_device < 0) return 2;
  backprojectionDDb(
    h_pVolume,
    h_pProj,
    h_pTubeAngle,
    h_pDetAngle,
    idXProj,
    nProj,
    nPixX, nPixY,
    nSlices,
    nDetX, nDetY,
    dx, dy, dz,
    du, dv,
    DSD, DDR, DAG);

  platform.finalize();
  double checkSum = 0;
  for (size_t i = 0; i < pixVol; i++)
    checkSum += h_pVolume[i];
  printf("checksum = %lf\n", checkSum);

  if (const char* output = getenv("HECBENCH_VOLUME_OUTPUT")) {
    FILE* fp = fopen(output, "wb");
    if (!fp || fwrite(h_pVolume, sizeof(double), pixVol, fp) != pixVol) return 2;
    fclose(fp);
  }
  free(h_pVolume);
  free(h_pTubeAngle);
  free(h_pDetAngle);
  free(h_pProj);
  return 0;
}
