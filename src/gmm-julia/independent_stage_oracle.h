// Independent mathematical stage checks, not the source CPU EM implementation.
// Full Gaussian identities in Float64, analytic 2x2 inverse, responsibility
// normalization, moments and covariance. Engineering budgets fixed pretrial.
struct GMMState {std::vector<float> N,pi,constant,avgvar,means,R,Rinv,memberships;};
static size_t oracle_checked=0;static double oracle_max_abs=0,oracle_max_rel=0;
static void gmm_near(double got,double want,double abs=2e-3,double rel=2e-4){
 if(!std::isfinite(got)||!std::isfinite(want)||std::abs(got-want)>abs+rel*std::abs(want)){
  fprintf(stderr,"INDEPENDENT_GMM_STAGE_MISMATCH check%zu got%.17g want%.17g budget%.17g\n",oracle_checked,got,want,abs+rel*std::abs(want));exit(8);
 }
 oracle_checked++;oracle_max_abs=std::max(oracle_max_abs,std::abs(got-want));oracle_max_rel=std::max(oracle_max_rel,std::abs(got-want)/std::max(1.0,std::abs(want)));
}
static void gmm_scientific_stage(int stage,int m,int d,int n,int gx,int gy,const std::vector<float>&data,const GMMState&in,const GMMState&out,const std::vector<float>&likelihood){
 if(d!=2||m<1||n<1){fprintf(stderr,"Unsupported named scientific fixture\n");exit(8);}
 if(stage==1){
  double mean[2]={0,0},v[2]={0,0};for(int e=0;e<n;e++)for(int k=0;k<2;k++){mean[k]+=data[2*e+k];v[k]+=double(data[2*e+k])*data[2*e+k];}
  double avg=0;for(int k=0;k<2;k++)avg+=v[k]/n-std::pow(mean[k]/n,2);avg/=2e6;
  for(int c=0;c<m;c++){
   float seed=m>1?(n-1.0f)/(m-1.0f):0.0f;int e=int(c*seed);
   for(int k=0;k<2;k++)gmm_near(out.means[c*2+k],data[2*e+k],0,0);
   gmm_near(out.N[c],double(n)/m);gmm_near(out.pi[c],1.0/m);gmm_near(out.avgvar[c],avg,1e-9,2e-4);
   for(int i=0;i<4;i++)gmm_near(out.R[c*4+i],(i==0||i==3)?1:0,0,0);
  }
 }else if(stage==2){
  double total=0;for(int c=0;c<m;c++)total+=in.N[c];
  for(int c=0;c<m;c++){
   double a=in.R[c*4],b=in.R[c*4+1],cc=in.R[c*4+2],dd=in.R[c*4+3],det=a*dd-b*cc;
   if(!(a>0&&dd>0&&det>0)){fprintf(stderr,"NON_SPD_GMM_COVARIANCE c%d\n",c);exit(8);}
   double inv[4]={dd/det,-b/det,-cc/det,a/det};for(int i=0;i<4;i++)gmm_near(out.Rinv[c*4+i],inv[i]);
   // Source PI is Float32; retain that source constant, not a pi-precision repair.
   gmm_near(out.constant[c],-std::log(2.0*double(PI))-0.5*std::log(det));
   gmm_near(out.pi[c],in.N[c]<.5?1e-10:in.N[c]/total,1e-7,2e-4);
  }
 }else if(stage==3){
  for(int c=0;c<m;c++)for(int e=0;e<n;e++){
   double x=data[e]-in.means[2*c],y=data[n+e]-in.means[2*c+1];
   double quad=x*x*in.Rinv[4*c]+x*y*in.Rinv[4*c+1]+y*x*in.Rinv[4*c+2]+y*y*in.Rinv[4*c+3];
   gmm_near(out.memberships[c*n+e],-.5*quad+in.constant[c]+std::log(double(in.pi[c])));
  }
 }else if(stage==4){
  std::vector<double> block(24,0);int step=(n/24)/16*16;
  for(int e=0;e<n;e++){
   double z=-INFINITY;for(int c=0;c<m;c++)z=std::max(z,double(in.memberships[c*n+e]));
   double sum=0;for(int c=0;c<m;c++)sum+=std::exp(double(in.memberships[c*n+e])-z);double norm=z+std::log(sum);double probs=0;
   for(int c=0;c<m;c++){double p=std::exp(double(in.memberships[c*n+e])-norm);gmm_near(out.memberships[c*n+e],p,1e-5,2e-4);probs+=out.memberships[c*n+e];}
   gmm_near(probs,1,2e-4,0);int b=step==0?23:std::min(e/step,23);block[b]+=norm;
  }
  for(int b=0;b<24;b++)gmm_near(likelihood[b],block[b]);
 }else if(stage==5){
  for(int c=0;c<m;c++){double count=0;for(int e=0;e<n;e++)count+=in.memberships[c*n+e];gmm_near(out.N[c],count);gmm_near(out.pi[c],count);}
 }else if(stage==6){
  for(int c=0;c<m;c++)for(int k=0;k<2;k++){double sum=0;for(int e=0;e<n;e++)sum+=double(data[k*n+e])*in.memberships[c*n+e];gmm_near(out.means[2*c+k],sum);}
 }else if(stage==7){
  for(int c=0;c<m;c++)for(int r=0;r<2;r++)for(int k=0;k<2;k++){
   double sum=0;for(int e=0;e<n;e++)sum+=(double(data[r*n+e])-in.means[2*c+r])*(double(data[k*n+e])-in.means[2*c+k])*in.memberships[c*n+e];
   if(r==k)sum+=in.avgvar[c];gmm_near(out.R[4*c+2*r+k],sum);
  }
 }
}
