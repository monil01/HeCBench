/*
GPU Implementation of Keccak by Guillaume Sevestre, 2010

This code is hereby put in the public domain.
It is given as is, without any guarantee.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#include "KeccakTreeCPU.h"
#include "KeccakTreeGPU.h"
#include "Test.h"
#include <iris/iris.hpp>


int main(int argc, char **argv)
{
   iris::Platform platform; platform.init(&argc,&argv,1);
   int count=0; iris_device_count(&count); bool gpu=false;
   for(int i=0;i<count;i++) {
     int b=0,t=0; char name[256]={}; size_t z;
     iris_device_info(i,iris_backend,&b,&z); iris_device_info(i,iris_type,&t,&z); iris_device_info(i,iris_name,name,&z);
     printf("IRIS backend%d/type%d %s\n",b,t,name); if(b==3&&t==256) gpu=true;
   }
   if(!gpu) return 2;
   Print_Param();
   TestCPU(1);
   TestGPU();
   Verify_results();
   platform.finalize();
   return 0;
}
