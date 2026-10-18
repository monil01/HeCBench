// CPU-only exact witness for original std::stof/std::stoi parsing and MNIST /255.0 policy.
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){std::ifstream fi(argv[1],std::ios::binary);std::vector<uint32_t>bits(4548580);fi.read((char*)bits.data(),bits.size()*4);std::ifstream manifest(argv[2]);std::string line;long checked=0;while(std::getline(manifest,line)){std::stringstream ss(line);std::string path;int sparse,H,rp,col,val;ss>>path>>sparse>>H>>rp>>col>>val;std::ifstream f(path);if(!f)return 2;std::vector<int>offset(H+1);int j=0;std::string a;while(std::getline(f,a)){float v;if(sparse){std::stringstream t(a);std::string row,c,w;std::getline(t,row,'\t');std::getline(t,c,'\t');std::getline(t,w,'\t');offset[std::stoi(row)+1]++;if((int)bits[col+j]!=std::stoi(c))return 3;checked++;v=std::stof(w);}else v=std::stof(a);uint32_t u;memcpy(&u,&v,4);if(bits[val+j]!=u){fprintf(stderr,"parse mismatch %s %d\n",path.c_str(),j);return 4;}checked++;j++;}if(sparse){for(int k=0;k<H;k++)offset[k+1]+=offset[k];for(int k=0;k<=H;k++){if((int)bits[rp+k]!=offset[k])return 5;checked++;}}}
 std::ifstream mn(argv[3],std::ios::binary);mn.seekg(16);for(int i=0;i<256*784;i++){unsigned char u;mn.read((char*)&u,1);float v=((float)u)/255.0;uint32_t b;memcpy(&b,&v,4);if(b!=bits[i])return 6;checked++;}
 printf("EXACT_SOURCE_INITIALIZER_VALUES %ld\n",checked);return 0;}
