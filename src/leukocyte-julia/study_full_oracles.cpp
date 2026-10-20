// Independent higher-precision equations; consumes saved actual native inputs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
template<class T> std::vector<T> read(std::ifstream& in,size_t n){std::vector<T>a(n);in.read((char*)a.data(),n*sizeof(T));if(!in)throw std::runtime_error("short input");return a;}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("detector input and flow input required");
 std::ifstream f(argv[1],std::ios::binary);auto h=read<int>(f,6);int m=h[0],n=h[1],nc=h[2],np=h[3],sm=h[4],sn=h[5];
 auto gx=read<float>(f,m*n),gy=read<float>(f,m*n),co=read<float>(f,np),si=read<float>(f,np);auto tx=read<int>(f,nc*np),ty=read<int>(f,nc*np);auto st=read<float>(f,sm*sn),gc=read<float>(f,m*n),dil=read<float>(f,m*n);
 size_t badg=0,badd=0,border=0;double maxg=0,maxgn=0;
 std::vector<double> samples(np);
 for(int x=0;x<n;x++)for(int y=0;y<m;y++){
  double best=0;
  if(x>=22&&x<n-22&&y>=22&&y<m-22){
   for(int k=0;k<nc;k++){
    double sum=0;for(int p=0;p<np;p++){int a=(x+tx[k*np+p])*m+y+ty[k*np+p];samples[p]=double(gx[a])*co[p]+double(gy[a])*si[p];sum+=samples[p];}
    double mean=sum/np,var=0;for(double p:samples)var+=(p-mean)*(p-mean);var/=(np-1);double score=mean*mean/var;if(score>best)best=score;
   }
  }else if(gc[x*m+y]!=0)border++;
  double err=std::abs(double(gc[x*m+y])-best),budget=1e-5+2e-4*std::abs(best);maxg=std::max(maxg,err);maxgn=std::max(maxgn,err/budget);if(!(err<=budget))badg++;
  float maximum=0;for(int i=0;i<sm;i++)for(int j=0;j<sn;j++){int yy=y-sm/2+i,xx=x-sn/2+j;if(yy>=0&&yy<m&&xx>=0&&xx<n&&st[i*sn+j]!=0)maximum=std::max(maximum,gc[xx*m+yy]);}if(dil[y*n+x]!=maximum)badd++;
 }
 printf("{\"detector_points\":%d,\"GICOV_failures\":%zu,\"GICOV_max_abs_error\":%.17g,\"GICOV_max_normalized_budget\":%.17g,\"border_failures\":%zu,\"dilation_failures\":%zu,\"flow_jobs\":[",m*n,badg,maxg,maxgn,border,badd);
 std::ifstream ff(argv[2],std::ios::binary);int jobs=read<int>(ff,1)[0];size_t badflow=0,points=0;double maxflow=0,maxfn=0;
 const double diffusion=double(float(.5f/5.f)), invpi=double(float(1.f/3.14159f)), inv_e=double(float(1.f/1e-10f));
 const int dr[]={-1,1,0,0,-1,1,-1,1},dc[]={0,0,-1,1,1,1,-1,-1},direction[]={-1,1,-1,1,0,2,-2,0};
 for(int job=0;job<jobs;job++){
  auto a=read<int>(ff,6);int rows=a[3],cols=a[4],steps=a[5],N=rows*cols;auto input=read<float>(ff,N),actual=read<float>(ff,N);std::vector<double>old(input.begin(),input.end()),next(N);
  for(int step=0;step<steps;step++){
   for(int r=0;r<rows;r++)for(int c=0;c<cols;c++){
    int p=r*cols+c;double sum=0,v=old[p];for(int j=0;j<8;j++){int rr=std::clamp(r+dr[j],0,rows-1),cc=std::clamp(c+dc[j],0,cols-1);double delta=old[rr*cols+cc]-v;double gate=std::atan((delta*direction[j])*inv_e)*invpi+.5;sum+=gate*delta;}
    double update=v+diffusion*sum;next[p]=update-(1.0/5.0)*double(input[p])*(update-double(input[p]));
   }old.swap(next);
  }
  double maxerr=0,maxnorm=0;size_t failures=0;for(int p=0;p<N;p++){double e=std::abs(double(actual[p])-old[p]),budget=1e-4+1e-4*std::abs(old[p]);maxerr=std::max(maxerr,e);maxnorm=std::max(maxnorm,e/budget);if(!(e<=budget))failures++;}
  if(job)printf(",");printf("{\"frames\":%d,\"frame\":%d,\"cell\":%d,\"m\":%d,\"n\":%d,\"iterations\":%d,\"points\":%d,\"failures\":%zu,\"max_abs_error\":%.17g,\"max_normalized_budget\":%.17g}",a[0],a[1],a[2],rows,cols,steps,N,failures,maxerr,maxnorm);
  points+=N;badflow+=failures;maxflow=std::max(maxflow,maxerr);maxfn=std::max(maxfn,maxnorm);
 }
 bool pass=!(badg||badd||border||badflow);printf("],\"flow_points\":%zu,\"flow_failures\":%zu,\"flow_max_abs_error\":%.17g,\"flow_max_normalized_budget\":%.17g,\"pass\":%s}\n",points,badflow,maxflow,maxfn,pass?"true":"false");return pass?0:1;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 2;}}
