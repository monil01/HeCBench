//******************************************************************
//cuSTSG is used to reconstruct high-quality NDVI time series data(MODIS/SPOT) based on STSG
//
//This procedure cuSTSG is the source code for the first version of cuSTSG.
//This is a parallel computing code using GPU.
//
//Coded by Yang Xue
// Reference:Xue Yang, Jin Chen, Qingfeng Guan, Huan Gao, and Wei Xia.
// Enhanced Spatial-Temporal Savitzky-Golay Method for Reconstructing High-quality NDVI Time Series: Reduced Sensitivity
// to Quality Flags and Improved Computational Efficiency.Transactions on Geoscience and Remote Sensing
//******************************************************************

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include "dmem.h"
#include <gdal/gdal_priv.h>


using namespace std;

int main(int argc, char *argv[])
{
  iris::Platform platform; platform.init(&argc,&argv,1);
  int dev=require_amd_hip();if(dev<0)return 2;iris_device_set_default(dev);
  GDALAllRegister();

  //parameters
  if (argc != 2)
  {
    cout << "No parameter file!" << endl;
    return 1;
  }

  ifstream parameter(argv[1]);
  if (!parameter)
  {
    cout << "Can't open parameter file!" << endl;
    return 1;
  }

  int* Years = nullptr;
  string NDVI_path, Reliability_path, STSG_Test_path;
  float cosyear, sampcorr;
  int win_year, win, snow_address, n_Years;
  string par;
  while (getline(parameter, par))
  {
    if (par.substr(0, 2) == "//" || par == "")
      continue;

    for (int i = 0; i < par.size(); )
    {
      if (isspace(par[i]))
        par.erase(i,1);
      else
        i++;
    }
    if (par.substr(0, par.find("=")) == "Years")
    {
      vector<int> year;
      while (par.rfind(",") < par.size())
      {
        year.push_back(stoi(par.substr(par.rfind(",") + 1)));
        par = par.substr(0, par.rfind(","));
      }
      year.push_back(stoi(par.substr(par.rfind("=") + 1)));

      n_Years = year.size();
      Years = new int[n_Years];
      for (int i = 0; i < n_Years; i++)
        Years[i] = year[n_Years - i - 1];
    }
    else if (par.substr(0, par.find("=")) == "NDVI_path")
      NDVI_path = par.substr(par.find("=") + 1);
    else if (par.substr(0, par.find("=")) == "Reliability_path")
      Reliability_path = par.substr(par.find("=") + 1);
    else if (par.substr(0, par.find("=")) == "STSG_Test_path")
      STSG_Test_path = par.substr(par.find("=") + 1);
    else if (par.substr(0, par.find("=")) == "cosyear")
      cosyear = stof(par.substr(par.find("=") + 1));
    else if (par.substr(0, par.find("=")) == "win_year")
      win_year = stoi(par.substr(par.find("=") + 1));
    else if (par.substr(0, par.find("=")) == "win")
      win = stoi(par.substr(par.find("=") + 1));
    else if (par.substr(0, par.find("=")) == "sampcorr")
      sampcorr = stof(par.substr(par.find("=") + 1));
    else if (par.substr(0, par.find("=")) == "snow_address")
      snow_address = stoi(par.substr(par.find("=") + 1));
  }
  parameter.close();

  char device_name[256]={};iris_device_info(dev,iris_name,device_name,nullptr);
  cout << "Device name:" << device_name << endl;
  size_t const totalGlobalMem = std::min(device_total_memory(), 4UL*1024*1024*1024);
  cout << "Device global memory used: " << totalGlobalMem / 1024 / 1024 << " MB" << endl;

  vector<GDALDataset*> NDVI(n_Years);
  vector<GDALDataset*> QA(n_Years);
  int n_X, n_Y, n_B;
  GDALDataType type_NDVI, type_QA;
  for (int i = 0; i < n_Years; i++)
  {
    string FileName = NDVI_path + to_string(Years[i]);
    NDVI[i] = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
    if (i == 0)
    {
      n_X = NDVI[i]->GetRasterXSize();
      n_Y = NDVI[i]->GetRasterYSize();
      n_B = NDVI[i]->GetRasterCount();
      type_NDVI = NDVI[i]->GetRasterBand(1)->GetRasterDataType();
    }

    FileName = Reliability_path + to_string(Years[i]);
    QA[i] = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
    if (i == 0)
      type_QA = QA[i]->GetRasterBand(1)->GetRasterDataType();
  }

  cout << "Execution start " << endl;
  size_t PerYSize = n_X*n_B *(n_Years * sizeof(short) + n_Years * sizeof(unsigned char) + 2 * n_Years * sizeof(float) + sizeof(int) + sizeof(float) + n_Years * sizeof(float)) + n_X*(2 * win + 1)*(2 * win + 1) *(7 * sizeof(float) + 3 * sizeof(int));
  if (totalGlobalMem <= 2 * win*n_X*n_B* (n_Years * sizeof(short) + n_Years * sizeof(unsigned char) + 2 * n_Years * sizeof(float) + sizeof(float)) + n_X*n_Y*n_B*n_Years*sizeof(float))
  {
    cout << "Size of vector_out is larger than total device global memory. Exit!" << endl;
    return 1;
  }

  size_t PerStep = (totalGlobalMem - 2 * win*n_X*n_B* (n_Years * sizeof(short) + n_Years * sizeof(unsigned char) + 2 * n_Years * sizeof(float) + sizeof(float)) - n_X*n_Y*n_B*n_Years*sizeof(float)) / PerYSize;
  int Loops = 1;
  if (PerStep < n_Y)
  {
    Loops = n_Y / PerStep + 1;
    PerStep = n_Y / Loops + 1;
  }

  float *d_vector_out;
  size_t nBytes = n_X*n_Y*n_B*n_Years * sizeof(float);
  allocateBuffer((void**)&d_vector_out, nBytes);
  resetBuffer((void*)d_vector_out, 0, nBytes);
  nBytes = win*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
  float *res = (float*)malloc(nBytes);
  memset((void*)res, 0, nBytes);
  int last_Buffer_Dn = 0;

  printf("Number of loops: %d\n", Loops);
  for (int i = 1, StartY = 0; i <= Loops && StartY < n_Y; i++, StartY += PerStep)
  {
    cout << "Loop " << i << endl;
    if (i == Loops)
      PerStep = n_Y - StartY;

    int Buffer_Up = 0;
    int Buffer_Dn = 0;
    if (StartY + PerStep < n_Y - win)
      Buffer_Dn = win;
    else
      Buffer_Dn = n_Y - PerStep - StartY;
    if (StartY >= win)
      Buffer_Up = win;
    else
      Buffer_Up = StartY;

    int blkwidth = 16;
    int blkheight = 16;
    dim3 blocks(blkwidth, blkheight);
    dim3 grids(n_X % blkwidth == 0 ? n_X / blkwidth : n_X / blkwidth + 1, (PerStep + Buffer_Up + Buffer_Dn) % blkheight == 0 ? (PerStep + Buffer_Up + Buffer_Dn) / blkheight : (PerStep + Buffer_Up + Buffer_Dn) / blkheight + 1);

    short *img_NDVI = new short[(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years];
    unsigned char *img_QA = new unsigned char[(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years];
    for (int i = 0; i < n_Years; i++)
    {
      NDVI[i]->RasterIO(GF_Read, 0, StartY - Buffer_Up, n_X, (PerStep + Buffer_Up + Buffer_Dn), &img_NDVI[i*(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B], n_X, (PerStep + Buffer_Up + Buffer_Dn), type_NDVI, n_B, nullptr, 0, 0, 0);
      QA[i]->RasterIO(GF_Read, 0, StartY - Buffer_Up, n_X, (PerStep + Buffer_Up + Buffer_Dn), &img_QA[i*(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B], n_X, (PerStep + Buffer_Up + Buffer_Dn), type_QA, n_B, nullptr, 0, 0, 0);
    }

    short *d_imgNDVI;
    nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years * sizeof(short);
    allocateBuffer((void**)&d_imgNDVI, nBytes);
    uploadBuffer((void*)d_imgNDVI, (void*)img_NDVI, nBytes);
    unsigned char *d_imgQA;
    nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years * sizeof(unsigned char);
    allocateBuffer((void**)&d_imgQA, nBytes);
    uploadBuffer((void*)d_imgQA, (void*)img_QA, nBytes);
    float *d_img_NDVI, *d_img_QA;
    nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years * sizeof(float);
    allocateBuffer((void**)&d_img_NDVI, nBytes);
    allocateBuffer((void**)&d_img_QA, nBytes);
    resetBuffer((void*)d_img_NDVI, 0, nBytes);
    resetBuffer((void*)d_img_QA, 0, nBytes);
    float *d_NDVI_Reference, *d_res;
    nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B * sizeof(float);
    allocateBuffer((void**)&d_NDVI_Reference, nBytes);
    resetBuffer((void*)d_NDVI_Reference, 0, nBytes);
    nBytes = (PerStep + Buffer_Dn)*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
    allocateBuffer((void**)&d_res, nBytes);
    resetBuffer((void*)d_res, 0, nBytes);
    nBytes = last_Buffer_Dn*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
    uploadBuffer((void*)d_res, (void*)res, nBytes);

    int *d_res_vec_res1;
    nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B * sizeof(int);
    allocateBuffer((void**)&d_res_vec_res1, nBytes);
    resetBuffer((void*)d_res_vec_res1, 0, nBytes);
    float *d_vector_in, *d_res_3;
    nBytes = PerStep*n_X* n_B * sizeof(float);
    allocateBuffer((void**)&d_vector_in, nBytes);
    resetBuffer((void*)d_vector_in, 0, nBytes);
    nBytes = PerStep*n_X*(2 * win + 1)*(2 * win + 1) * 3 * sizeof(float);
    allocateBuffer((void**)&d_res_3, nBytes);//(slope_intercept(2);corr_similar;)
    resetBuffer((void*)d_res_3, 0, nBytes);
    int *d_index;
    nBytes = PerStep*n_X*(2 * win + 1)*(2 * win + 1) * 3 * sizeof(int);
    allocateBuffer((void**)&d_index, nBytes);//(similar_index(2);new_corr;)
    resetBuffer((void*)d_index, 0, nBytes);
    // IRIS task submits synchronously.

    auto start = std::chrono::steady_clock::now();
    submitKernel("Short_to_Float",grids,blocks,{memory(d_imgNDVI),memory(d_imgQA),scalar(static_cast<int>(n_X)),scalar(static_cast<int>((PerStep + Buffer_Up + Buffer_Dn))),scalar(static_cast<int>(n_B)),scalar(static_cast<int>(n_Years)),memory(d_img_NDVI),memory(d_img_QA)});

    submitKernel("Generate_NDVI_reference",grids,blocks,{scalar(static_cast<float>(cosyear)),scalar(static_cast<int>(win_year)),memory(d_img_NDVI),memory(d_img_QA),scalar(static_cast<int>(n_X)),scalar(static_cast<int>((PerStep + Buffer_Up + Buffer_Dn))),scalar(static_cast<int>(n_B)),scalar(static_cast<int>(n_Years)),memory(d_NDVI_Reference),memory(d_res_3),memory(d_res_vec_res1)});

    nBytes = PerStep*n_X*(2 * win + 1)*(2 * win + 1) * 3 * sizeof(float);
    resetBuffer((void*)d_res_3, 0, nBytes);

    submitKernel("Compute_d_res",grids,blocks,{memory(d_img_NDVI),memory(d_img_QA),memory(d_NDVI_Reference),scalar(static_cast<int>(StartY)),scalar(static_cast<int>(n_Y)),scalar(static_cast<int>(Buffer_Up)),scalar(static_cast<int>(Buffer_Dn)),scalar(static_cast<int>(n_X)),scalar(static_cast<int>((PerStep + Buffer_Up + Buffer_Dn))),scalar(static_cast<int>(n_B)),scalar(static_cast<int>(n_Years)),scalar(static_cast<int>(win)),memory(d_res)});

    submitKernel("STSG_filter",grids,blocks,{memory(d_img_NDVI),memory(d_img_QA),memory(d_NDVI_Reference),scalar(static_cast<int>(StartY)),scalar(static_cast<int>(n_Y)),scalar(static_cast<int>(Buffer_Up)),scalar(static_cast<int>(Buffer_Dn)),scalar(static_cast<int>(n_X)),scalar(static_cast<int>(PerStep)),scalar(static_cast<int>(n_B)),scalar(static_cast<int>(n_Years)),scalar(static_cast<int>(win)),scalar(static_cast<float>(sampcorr)),scalar(static_cast<int>(snow_address)),memory(d_vector_out),memory(d_vector_in),memory(d_res),memory(d_res_3),memory(d_index)});

    // IRIS task submits synchronously.
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    cout << "Total kernel time: " << time * 1e-9f  << " s" << endl;

    nBytes = win*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
    memset((void*)res, 0, nBytes);
    nBytes = Buffer_Dn*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
    downloadBuffer((void*)res, (void*)&d_res[(PerStep + Buffer_Dn - win)*n_X*(2 * win + 1)*(2 * win + 1) * 4], nBytes);
    last_Buffer_Dn = Buffer_Dn;

    delete[] img_NDVI;
    delete[] img_QA;
    releaseBuffer((void*)d_imgNDVI);
    releaseBuffer((void*)d_imgQA);
    releaseBuffer((void*)d_img_NDVI);
    releaseBuffer((void*)d_img_QA);
    releaseBuffer((void*)d_NDVI_Reference);
    releaseBuffer((void*)d_res);
    releaseBuffer((void*)d_res_vec_res1);
    releaseBuffer((void*)d_vector_in);
    releaseBuffer((void*)d_res_3);
    releaseBuffer((void*)d_index);
  }
  free((void*)res);

  float *vector_out = new float[n_X*n_Y*n_B*n_Years];
  nBytes = n_X*n_Y*n_B*n_Years* sizeof(float);
  downloadBuffer((void*)vector_out, (void*)d_vector_out, nBytes);
  releaseBuffer((void*)d_vector_out);

  if (const char* path=std::getenv("HECBENCH_OUTPUT")) {
    FILE* fp=std::fopen(path,"wb");if(!fp)std::abort();
    std::fwrite(vector_out,sizeof(float),n_X*n_Y*n_B*n_Years,fp);std::fclose(fp);
  }
  long cnt = 0;
  double sum = 0;
  for (int i = 0; i < n_X*n_Y*n_B*n_Years; i++) {
    if (vector_out[i] < 1.f || vector_out[i] > 0.f) {
      sum += vector_out[i];
      cnt++;
    }
  }
  cout << "Checksum: " << sum << " " << cnt << " " << sum / cnt << endl;

  delete[] vector_out;
  platform.finalize();

  return 0;
}
