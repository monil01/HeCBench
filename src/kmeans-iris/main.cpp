#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "kmeans.h"
#include "dmem.h"

int main( int argc, char** argv) 
{
  iris::Platform platform;if(platform.init(&argc,&argv,true)!=IRIS_SUCCESS)return 1;
  int dev=require_amd_hip();if(dev<0)return 1;iris_device_set_default(dev);
	printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d \n", BLOCK_SIZE, BLOCK_SIZE2);

	setup(argc, argv);

	int errors=platform.error_count();platform.finalize();return errors?1:0;
}
