#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include "device.h"

int CorMat_singlePass(float* , float * , int , int);
int CorMat_multiPass(float* , float * , int , int);

size_t remaining_B(int N, size_t available_memory)
{
  size_t x=available_memory;
  size_t temp=N;
  temp*=2;
  x/=temp;
  return x;
}

void preprocessing(float * data_t, int N,int L)
{
  for (int i = 0; i < N; i++)
  {
    float * row = data_t + i * L;
    double sum1 = 0, sum2 = 0;
    for (int l = 0; l < L; l++)
    {
      sum1 += row[l];
    }
    sum1 /= L;
    for (int l = 0; l < L; l++)
    {
      sum2 += (row[l] - sum1) * (row[l] - sum1);
    }
    sum2 = sqrt(sum2);
    for (int l = 0; l < L; l++)
    {
      if(sum2!=0)
        row[l] = (row[l] - sum1) / sum2;
      else
        if(sum2==0)
          row[l]=0;
    }
  }
}



extern "C" void study_preprocessing(float*data,int N,int L){preprocessing(data,N,L);}
