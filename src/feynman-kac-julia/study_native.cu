#define main legacy_main
#include "feynman-kac-cuda/main.cu"
#undef main
#include <cmath>
double host_rng(int*s){int k=*s/127773;*s=16807*(*s-k*127773)-k*2836;if(*s<0)*s+=2147483647;return double(*s)*4.656612875e-10;}
double host_potential(double a,double b,double x,double y){return 2*(pow(x/a/a,2.)+pow(y/b/b,2.))+1/a/a+1/b/b;}
void cpu_point (
    const int cpu_i,const int cpu_j,const int ni,
    const int nj,
          int seed,
    const int N,
    const double a,
    const double b,
    const double h,
    const double rth,
    int *__restrict__ n_inside,
    double *__restrict__ err)
{
  int i = cpu_i;
  int j = cpu_j;
  if (i <= ni && j <= nj) {
    double x = ( ( double ) ( nj - j     ) * ( - a )
               + ( double ) (      j - 1 ) *     a )
               / ( double ) ( nj     - 1 );

    double y = ( ( double ) ( ni - i     ) * ( - b )
               + ( double ) (      i - 1 ) *     b ) 
               / ( double ) ( ni     - 1 );

    double dx;
    double dy;
    double us;
    double ut;
    double vh;
    double vs;
    double x1;
    double x2;
    double w;
    double w_exact;
    double we;
    double wt;
    double chk = pow ( x / a, 2.0 ) + pow ( y / b, 2.0 );

    if ( 1.0 < chk )
    {
      w_exact = 1.0;
      wt = 1.0;
    }
    else {
      *n_inside += 1;
      w_exact = exp ( pow ( x / a, 2.0 ) + pow ( y / b, 2.0 ) - 1.0 );
      wt = 0.0;
      for ( int k = 0; k < N; k++ )
      {
        x1 = x;
        x2 = y;
        w = 1.0;  
        chk = 0.0;
        while ( chk < 1.0 )
        {
          ut = host_rng ( &seed );
          if ( ut < 1.0 / 2.0 )
          {
            us = host_rng ( &seed ) - 0.5;
            if ( us < 0.0)
              dx = - rth;
            else
              dx = rth;
          } 
          else
          {
            dx = 0.0;
          }

          ut = host_rng ( &seed );
          if ( ut < 1.0 / 2.0 )
          {
            us = host_rng ( &seed ) - 0.5;
            if ( us < 0.0 )
              dy = - rth;
            else
              dy = rth;
          }
          else
          {
            dy = 0.0;
          }
          vs = host_potential ( a, b, x1, x2 );
          x1 = x1 + dx;
          x2 = x2 + dy;

          vh = host_potential ( a, b, x1, x2 );

          we = ( 1.0 - h * vs ) * w;
          w = w - 0.5 * h * ( vh * we + vs * w ); 

          chk = pow ( x1 / a, 2.0 ) + pow ( x2 / b, 2.0 );
        }
        wt += w;
      }
      wt /= ( double ) ( N ); 
      *err += pow(w_exact-wt,2.0);
    }
  }
}

int main(){const int ni=17,nj=9,seed=123456789,paths=32;const double a=2,b=1,h=.001,rth=sqrt(2*h);int want_n=0;double want_e=0;for(int j=1;j<=nj;j++)for(int i=1;i<=ni;i++)cpu_point(i,j,ni,nj,seed,paths,a,b,h,rth,&want_n,&want_e);int*dn;double*de;cudaMalloc(&dn,4);cudaMalloc(&de,8);auto reset=[&](){cudaMemset(dn,0,4);cudaMemset(de,0,8);cudaDeviceSynchronize();};auto run=[&](){fk<<<dim3((ni+15)/16,(nj+15)/16),dim3(16,16)>>>(ni,nj,seed,paths,a,b,h,rth,dn,de);};reset();run();if(cudaDeviceSynchronize()!=cudaSuccess)return 2;for(int trial=1;trial<=5;trial++){double time=0;for(int k=0;k<3;k++){reset();auto t=std::chrono::steady_clock::now();run();if(cudaDeviceSynchronize()!=cudaSuccess)return 2;time+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/3;}int n;double e;cudaMemcpy(&n,dn,4,cudaMemcpyDeviceToHost);cudaMemcpy(&e,de,8,cudaMemcpyDeviceToHost);double err=fabs(e-want_e),out[2]={double(n),e};bool ok=n==want_n&&std::isfinite(e)&&err<=1e-8+1e-8*fabs(want_e);char path[64];snprintf(path,sizeof(path),"trial_%d.bin",trial);FILE*f=fopen(path,"wb");fwrite(out,8,2,f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.15g,\"correct\":%s,\"checked_values\":2,\"max_abs_error\":%.15g,\"timing_region\":\"mean_stochastic_kernel_excluding_reset\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":2,\"byte_order\":\"little\"}}\n",trial,time,ok?"true":"false",err,path);if(!ok)return 1;}cudaFree(dn);cudaFree(de);}
