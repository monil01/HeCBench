// One-step GPU unit on an authentic captured cell; never replaces video input.
#include "find_ellipse.h"
#include <cstdio>
#include <vector>
#ifdef IRIS_REFERENCE_DRIVER
#include "iris_helpers.h"
#endif
extern void IMGVF_GPU(MAT**,MAT**,double,double,double,int,double,int);
int main(int argc,char**argv){
 if(argc!=3)return 1;
#ifdef IRIS_REFERENCE_DRIVER
 iris::Platform platform;iris_check(platform.init(&argc,&argv,1));require_hip_gpu();
#endif
 FILE*f=fopen(argv[1],"rb");if(!f)return 1;
 int shape[3];if(fread(shape,sizeof(int),3,f)!=3)return 1;
 std::vector<int>dims(3*shape[1]);if(fread(dims.data(),sizeof(int),dims.size(),f)!=dims.size())return 1;
 std::vector<float>in(shape[2]);if(fread(in.data(),sizeof(float),in.size(),f)!=in.size())return 1;fclose(f);
 int m=dims[shape[1]],n=dims[2*shape[1]],offset=dims[0];
 MAT*image=m_get(m,n),*out=m_get(m,n);
 for(int i=0;i<m;++i)for(int j=0;j<n;++j)m_set_val(image,i,j,in[offset+i*n+j]);
 MAT*images[]={image};MAT*outputs[]={out};
 IMGVF_GPU(images,outputs,1.,1.,1e-10,1,0.,1);
 f=fopen(argv[2],"wb");if(!f)return 1;
 for(int i=0;i<m;++i)for(int j=0;j<n;++j){float x=m_get_val(out,i,j);if(fwrite(&x,sizeof(x),1,f)!=1)return 1;}
 fclose(f);m_free(out);m_free(image);
#ifdef IRIS_REFERENCE_DRIVER
 iris_check(platform.finalize());
#endif
}
