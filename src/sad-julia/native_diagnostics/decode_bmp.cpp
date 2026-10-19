#include <fstream>
#include <vector>
#include "../original_assets/bitmap_image.hpp"
int main(int argc,char**argv){if(argc!=3)return 1;bitmap_image bmp(argv[1]);if(bmp.width()==0||bmp.height()==0)return 2;std::vector<unsigned char> a;for(unsigned y=0;y<bmp.height();y++)for(unsigned x=0;x<bmp.width();x++){rgb_t c;bmp.get_pixel(x,y,c);a.push_back(c.red);a.push_back(c.green);a.push_back(c.blue);}std::ofstream f(argv[2],std::ios::binary);f.write((char*)a.data(),a.size());return f?0:3;}
