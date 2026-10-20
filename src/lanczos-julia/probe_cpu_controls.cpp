#include <cstdio>
#include <fstream>
#include <iomanip>
#include "eigen.h"
#include "graph_controls.h"
int main(int argc,char**argv){for(int c=0;c<5;c++){
 auto coo=control_graph(c);csr_matrix<double> m(coo);auto e=lanczos_eigen(m,3,71);
 printf("CONTROL_EIGEN %d %d %d",c,m.row_size(),m.nonzeros());for(double v:e)printf(" %.17g",v);printf("\n");
 if(argc>1){char f[1024];sprintf(f,"%s/control%d_csr.bin",argv[1],c);FILE*out=fopen(f,"wb");int h[3]={m.row_size(),m.nonzeros(),int(e.size())};fwrite(h,4,3,out);fwrite(m.row_ptr_data(),4,m.row_size()+1,out);fwrite(m.col_ind_data(),4,m.nonzeros(),out);fwrite(m.values_data(),8,m.nonzeros(),out);fwrite(e.data(),8,e.size(),out);fclose(out);}
}}
