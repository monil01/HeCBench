#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../winograd-cuda/utils.h"
int main(){std::vector<float>data;float filter[16];WinogradConv2D_2x2_filter_transformation(filter);for(int c=0;c<2;c++){for(auto x:filter)data.push_back(x);srand(1);for(int x=0;x<131;x++)for(int y=0;y<131;y++)data.push_back(c==0 ? rand()/(float)RAND_MAX : ((x*17+y*23)%257-128)/128.f);}FILE*f=fopen("fixture_f32.bin","wb");fwrite(data.data(),4,data.size(),f);fclose(f);}
