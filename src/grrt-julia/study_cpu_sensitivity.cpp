// CPU-only controlled arithmetic/controller and plane-event diagnostics; not primary solver changes.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include "../grrt-cuda/constants.h"
#undef IMAGE_SIZE
#define IMAGE_SIZE 8

namespace d12 {
static double cpu_event[64][3]={};
// Deterministic host-only clone of original CUDA source; qualifiers/index mapping changed, diagnostics added without changing equations.
static int cpu_x,cpu_y; static long long cpu_accepted_steps=0,cpu_shrunk_steps=0,cpu_transfer_steps=0,cpu_transfer_rejected_steps=0; static double cpu_initial_null_max=0;
/*
  Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
  Odyssey  version 1.0   (released  2015)
  This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
  for General Relativistic Radiative Transfer (GRRT), following the 
  ray-tracing algorithm presented in 
  Fuerst, S. V., & Wu, K. 2007, A&A, 474, 55, 
  and the radiative transfer formulation described in 
  Younsi, Z., Wu, K., & Fuerst, S. V. 2012, A&A, 545, A13

  Odyssey is distributed freely under the GNU general public license. 
  You can redistribute it and/or modify it under the terms of the License
  http://www.gnu.org/licenses/gpl.txt
  The current distribution website is:
  https://github.com/hungyipu/Odyssey/ 
 */


 static void cpu_geodesic(double* Variables, double* VariablesIn, double *y, double *dydx)
{
  double r = y[0];
  double theta = y[1];
  double pr = y[4];
  double ptheta = y[5];

  double r2  = r * r;
  double twor = 2.0 * r;

  double sintheta = sin(theta);
  double costheta = cos(theta);
  double cos2     = costheta * costheta;
  double sin2     = sintheta * sintheta;
  double sigma  = r2 + a2 * cos2;
  double delta  = r2 - twor + a2;
  double sd    = sigma * delta;
  double siginv  = 1.0 / sigma;
  double bot    = 1.0 / sd;


  //avoid problems near the axis
  if (sintheta < 1e-8)
  {
    sintheta = 1e-8;
    sin2 = 1e-16;
  }

  dydx[0] = -pr * delta * siginv;  
  dydx[1] = -ptheta * siginv;
  dydx[2] = -(twor * A + (sigma - twor) * L / sin2) * bot;
  dydx[3] = -(1.0 + (twor * (r2 + a2) - twor * A * L) * bot);
  dydx[4] = -(((r - 1.0) * (-kappa) + twor * (r2 + a2) - 2.0 * A * L) * bot - 2.0 * pr * pr * (r - 1.0) * siginv);
  dydx[5] = -sintheta * costheta*(L * L / (sin2 * sin2) - a2) * siginv;
}

 static void cpu_rkstep(double* Variables, double* VariablesIn,double *y,
                              double *dydx, double h, double *yout, double *yerr)
{
  int i;
  double ak[N];
  double ytemp1[N], ytemp2[N], ytemp3[N], ytemp4[N], ytemp5[N];
  double hdydx;
  double yi, yt;

  for (i = 0; i < N; i++)
  {
    hdydx     = h * dydx[i];
    yi        = y[i];
    ytemp1[i] = yi + 0.2 * hdydx;
    ytemp2[i] = yi + (3.0/40.0) * hdydx;
    ytemp3[i] = yi + 0.3 * hdydx;
    ytemp4[i] = yi -(11.0/54.0) * hdydx;
    ytemp5[i] = yi + (1631.0/55296.0) * hdydx;
    yout[i]   = yi + (37.0/378.0) * hdydx;
    yerr[i]   = ((37.0/378.0)-(2825.0/27648.0)) * hdydx;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp1, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp2[i] += (9.0/40.0) * yt;
    ytemp3[i] -= 0.9 * yt;
    ytemp4[i] += 2.5 * yt;
    ytemp5[i] += (175.0/512.0) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp2, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp3[i] += 1.2 * yt;
    ytemp4[i] -= (70.0/27.0) * yt;
    ytemp5[i] += (575.0/13824.0) * yt;
    yout[i]   += (250.0/621.0) * yt;
    yerr[i]   += ((250.0/621.0)-(18575.0/48384.0)) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp3, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp4[i] += (35.0/27.0) * yt;
    ytemp5[i] += (44275.0/110592.0) * yt;
    yout[i]   += (125.0/594.0) * yt;
    yerr[i]   += ((125.0/594.0)-(13525.0/55296.0)) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp4, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp5[i] += (253.0/4096.0) * yt;
    yerr[i]   -= (277.0/14336.0) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp5, ak);

  for (i = 0; i < N; i++)
  {
    yt       = h * ak[i];
    yout[i] += (512.0/1771.0) * yt;
    yerr[i] += ((512.0/1771.0)-0.25) * yt;
  }
}

 static double cpu_rk5(double* Variables, double* VariablesIn, double *y, double *dydx, 
                             double htry, double escal, double *yscal, double *hdid)
{
  int i;
  double hnext;
  double errmax, h = htry, htemp;
  double yerr[N], ytemp[N];

  while (1)
  {
    // find adaptive step size
    cpu_rkstep(Variables, VariablesIn, y, dydx, h, ytemp, yerr);

    errmax = 0.0;
    for (i = 0; i < N; i++)
    {
      double temp = fabs(yerr[i]/yscal[i]);
      if (temp > errmax) errmax = temp;
    }

    errmax *= escal;
    if (errmax <= 1.0) break;

    htemp = 0.9 * h / sqrt(sqrt(errmax));

    h *= 0.1;

    if (h >= 0.0)
    {
      if (htemp > h) h = htemp;
    }
    else
    {
      if (htemp < h) h = htemp;
    }
  }

  if (errmax > 1.89e-4)
  {
    hnext = 0.9 * h * pow(errmax, -0.2);
  }
  else
  {
    hnext = 5.0 * h;
  }

  *hdid = h; cpu_accepted_steps++; if(h != htry) cpu_shrunk_steps++;

  memcpy(y, ytemp, N * sizeof(double));

  return hnext;
}

 static void cpu_initial(double* Variables, double* VariablesIn, double *y0, double *ydot0)
{
  double alpha = grid_x;
  double beta  = grid_y;

  //see Eq[18-19], with phi_obs=0, x= alpha, y=beta, z=0
  double x     = sqrt(r0*r0+a2)*sin(theta0)-beta*cos(theta0);
  double y     = alpha;
  double z     = r0*cos(theta0)+beta*sin(theta0);
  double w     = x*x+y*y+z*z-a2;

  //see Eq[20-22]
  y0[0] = sqrt((w+sqrt(w*w+(2.*A*z)*(2.*A*z)))/2.);   
  y0[1] = acos(z/y0[0]);                              
  y0[2] = atan2(y,x);                                
  y0[3] = 0;
  double r = y0[0];
  double theta = y0[1];
  double phi=y0[2];

  double sigma = r*r+(A*cos(theta))*(A*cos(theta));
  double u     = sqrt(a2+r*r);
  double v     = -sin(theta0)*cos(phi);
  double zdot  = -1.;

  //see Eq[24-26]
  double rdot0 = zdot*(-u*u*cos(theta0)*cos(theta)+r*u*v*sin(theta))/sigma;         
  double thetadot0 = zdot*(cos(theta0)*r*sin(theta)+u*v*cos(theta))/sigma;          
  double phidot0 = zdot*sin(theta0)*sin(phi)/(u*sin(theta));                         

  ydot0[0] = rdot0;
  ydot0[1] = thetadot0;
  ydot0[2] = phidot0;

  double sintheta=sin(theta);
  double sin2 = sintheta*sintheta;

  double r2 = r * r;
  double delta = r2 - 2.0 * r + a2;
  double s1 = sigma - 2.0 * r;

  y0[4] = rdot0*sigma/delta;
  y0[5] = thetadot0*sigma;

  // Eq.7
  double energy2 = s1*(rdot0*rdot0/delta+thetadot0*thetadot0)
    + delta*sin2*phidot0*phidot0;

  double energy = sqrt(energy2);

  // rescaled by energy
  y0[4] = y0[4]/energy;
  y0[5] = y0[5]/energy;

  // Eq.8 
  L = ((sigma*delta*phidot0-2.0*A*r*energy)*sin2/s1)/energy;

  // Eq below Eq.15
  kappa = y0[5]*y0[5]+a2*sin2+L*L/sin2; if(A==0){double H=-1/(1-2/r)+(1-2/r)*y0[4]*y0[4]+y0[5]*y0[5]/r2+L*L/(r2*sin2);cpu_initial_null_max=fmax(cpu_initial_null_max,fabs(H));}
}

 static float cpu_ISCO(double* VariablesIn)
{
  double z1       = 1 + pow(1 - A * A, 1 / 3.0) * pow(1 + A, 1 / 3.0) + pow(1 - A, 1 / 3.0);
  double z2       = sqrt(3 * A * A + z1 * z1);
  return 3. + z2 - sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

//
// functions for radiative transfer                                                  
//

#define Te_min    0.1
#define Te_max    100.
#define Te_grids  50.

static   double cpu_K2_tab[] = {
  -10.747001, //Te=0.1
  -9.362569,
  -8.141373,
  -7.061568,
  -6.104060,
  -5.252153,
  -4.491244,
  -3.808555,
  -3.192909,
  -2.634534,
  -2.124893,
  -1.656543,
  -1.223007,
  -0.818668,
  -0.438676,
  -0.078863,
  +0.264332,
  +0.593930,
  +0.912476,
  +1.222098,
  +1.524560,
  +1.821311,
  +2.113537,
  +2.402193,
  +2.688050,
  +2.971721,
  +3.253692,
  +3.534347,
  +3.813984,
  +4.092839,
  +4.371092,
  +4.648884,
  +4.926323,
  +5.203493,
  +5.480457,
  +5.757264,
  +6.033952,
  +6.310550,
  +6.587078,
  +6.863554,
  +7.139990,
  +7.416395,
  +7.692778,
  +7.969143,
  +8.245495,
  +8.521837,
  +8.798171,
  +9.074500,
  +9.350824,
  +9.627144  //Te=87.09
};

 static double cpu_K2_find(double Te)
{
  double d = Te_grids*(log(Te / Te_min)/ log(Te_max / Te_min));
  int    i = floor(d);

  return (1 - (double)(d-i)) * cpu_K2_tab[i] + (double)(d-i) * cpu_K2_tab[i+1];
}

 static double cpu_K2(double Te)
{
  double tab_K2;
  //avoid the boundary effect when T~T_max
  if (Te>85.){ 
    tab_K2=2.*Te*Te;
    return tab_K2;
  }

  if (Te<Te_min){ 
    //set a dummy value
    return exp(-11.);
  }

  tab_K2= cpu_K2_find(Te);
  return exp(tab_K2);
}

 static double cpu_Jansky_Correction(double* VariablesIn,double ima_width)
{
  double distance=C_sgrA_d*C_pc;
  double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
  double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //radian^2 (steradians) per pixel
  return pix_str/C_Jansky;
}

 static double cpu_Luminosity_Correction(double* VariablesIn,double ima_width)
{
  double distance=C_sgrA_d*C_pc;
  double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
  double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //radian^2 (steradians) per pixel
  return pix_str*distance*distance*4.*PI*freq_obs;
}

 double cpu_task1fun_GetZ(double* Variables, double* VariablesIn, double *y)
{
  double r1 = y[0];
  double E_local = -(r1 * r1 + A * sqrt(r1)) / (r1 * sqrt(r1 * r1 - 3. * r1 + 2. * A * sqrt(r1))) + 
                  L / sqrt(r1) / sqrt(r1 * r1 - 3. * r1  +2. * A * sqrt(r1));
  double E_inf = -1.0;      
  return E_local / E_inf; 
}

 void cpu_task1(double*__restrict__ ResultsPixel,
                      double*__restrict__ VariablesIn,
                      int GridIdxX, int GridIdxY)
{
  // to check whether the photon is inside image plane
  if(cpu_x >= SIZE || cpu_y >= SIZE) return;  

  double Variables[VarNUM];

  r0       = 1000.0;                 
  theta0   = (PI/180.0) * INCLINATION;     
  a2       = A * A;                
  Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;      
  Rmstable = cpu_ISCO(VariablesIn);                

  double htry = 0.5, escal = 1e12, hdid = 0.0, hnext = 0.0;
  double y[N], dydx[N], yscal[N], ylaststep[N];
  double Rdisk   = 50.;
  double ima_width = 55.;
  double s1  = ima_width;                      
  double s2  = 2.*ima_width/((int)SIZE+1.);   

  grid_x = -s1 + s2*(cpu_x+1.);
  grid_y = -s1 + s2*(cpu_y+1.);

  cpu_initial(Variables, VariablesIn, y, dydx);

  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+0] = grid_x;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+1] = grid_y;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 0;

  while (1)
  {
    for(int i = 0; i < N; i++) ylaststep[i] = y[i];

    cpu_geodesic(Variables, VariablesIn, y, dydx);

    for (int i = 0; i < N; i++) yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

    //fifth-order Runge-Kutta method
    hnext = cpu_rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

    // hit the disk, compute redshift
    if( y[0] < Rdisk && y[0] > Rmstable && (ylaststep[1] - PI/2.) * (y[1] - PI/2.) < 0. )
    {    
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 1./cpu_task1fun_GetZ(Variables, VariablesIn, y);
      double fraction=(PI/2.-ylaststep[1])/(y[1]-ylaststep[1]);
      double plane[N];for(int q=0;q<N;q++)plane[q]=ylaststep[q]+fraction*(y[q]-ylaststep[q]);
      int pix=IMAGE_SIZE*cpu_y+cpu_x;
      cpu_event[pix][0]=ResultsPixel[3*pix+2];
      cpu_event[pix][1]=1./cpu_task1fun_GetZ(Variables,VariablesIn,plane);
      cpu_event[pix][2]=fabs(1./cpu_task1fun_GetZ(Variables,VariablesIn,ylaststep)-cpu_event[pix][0]);

      break;
    }

    // Inside the event horizon radius or escape to infinity
    if ((y[0] > r0) && (dydx[0]>0)) break;

    if (y[0] < Rhor) break;

    htry = hnext;
  }
}

 double cpu_task2fun_GetZ(double* Variables, double* VariablesIn, double *y)
{
  double ut,uphi,ur,E_local;
  double E_inf= -1.0;   
  double r=y[0];
  double theta=y[1];
  double pr=y[4];

  double r2 = r*r;
  double twor = 2.0*r;
  double sintheta, costheta;
  sintheta=sin(theta);
  costheta=cos(theta);
  double cos2 = costheta*costheta;
  double sin2 = sintheta*sintheta;

  double sigma = r2+a2*cos2;
  double delta = r2-twor+a2;
  double ssig=(r2+a2)*(r2+a2)-a2*delta*sin2; 

  //======define (covariant) metric component    
  double gtt=-(1.-2.*r/sigma);
  double gtph=-2.*A*r*sin2/sigma;
  double grr=sigma/delta;
  double gphph=ssig*sin2/sigma;    


  //==========Keplerian flow: outside cpu_ISCO
  double ut_k  =(r*r+A*sqrt(r))/(r*sqrt(r*r-3.*r+2.*A*sqrt(r)));
  double ur_k  =0.;
  double uphi_k  =1./(sqrt(r)*sqrt(r*r-3.*r+2.*A*sqrt(r)));

  //==========Keplerian flow: inside cpu_ISCO    
  if( r<Rmstable)
  {

    double delta = r*r-2.*r+a2;
    double lambda=(Rmstable*Rmstable-2.*A*sqrt(Rmstable)+a2)/(sqrt(Rmstable*Rmstable*Rmstable)-2.*sqrt(Rmstable)+A);
    double gamma=sqrt(1-2./3./Rmstable);
    double h=(2.*r-A*lambda)/delta;

    ut_k=gamma*(1.+2/r*(1.+h));
    ur_k=-sqrt(2./3./Rmstable)*sqrt(pow((Rmstable/r-1.),3.));
    uphi_k=gamma/r/r*(lambda+A*h);
  }  

  //normalize the four-velocity
  ut    = ut_k;
  uphi  = uphi_k;
  ur    = ur_k;  
  double omega = uphi/ut;
  double k0    = -(gtt + omega*omega*gphph + 2.*omega*gtph);
  ut = sqrt(((1. + grr*ur*ur) / k0));
  uphi = omega*ut;  

  // compute redshift
  E_local=-ut+L*uphi+pr*ur;
  return E_local/E_inf; 
}

 void cpu_task2(double*__restrict__ ResultsPixel,
                      double*__restrict__ VariablesIn,
                      int GridIdxX, int GridIdxY)
{
  if(cpu_x >= SIZE || cpu_y >= SIZE) return;  

  double Variables[VarNUM];
  r0       = 1000.0;                 
  theta0   = (PI/180.0) * INCLINATION;     
  a2       = A * A;                
  Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;   
  Rmstable = cpu_ISCO(VariablesIn);  

  double htry = 0.5, escal = 1e12, hdid = 0.0, hnext = 0.0;
  double y[N], dydx[N], yscal[N];

  double Rdisk   = 500.;
  double ima_width = 10.;
  double s1  = ima_width;                      
  double s2  = 2.*ima_width/((int)SIZE+1.);   
  double Jy_corr=cpu_Jansky_Correction(VariablesIn,ima_width);
  double L_corr=cpu_Luminosity_Correction(VariablesIn,ima_width);

  grid_x = -s1 + s2*(cpu_x+1.);
  grid_y = -s1 + s2*(cpu_y+1.);

  cpu_initial(Variables, VariablesIn, y, dydx);

  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+0] = grid_x;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+1] = grid_y;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 0;

  double ds=0.;  
  double dtau=0.;
  double dI=0.;

  while (1)
  {
    cpu_geodesic(Variables, VariablesIn, y, dydx);

    for (int i = 0; i < N; i++)
      yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

    hnext = cpu_rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

    if ((y[0] > r0) && (dydx[0]>0)){
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = dI*freq_obs*freq_obs*freq_obs*L_corr; 
      break;
    }

    if (y[0] < Rhor){
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = dI*freq_obs*freq_obs*freq_obs*L_corr; 
      break;
    }

    double r=y[0];
    double theta=y[1];

    if(y[0]<Rdisk){

      double zzz        = cpu_task2fun_GetZ(Variables, VariablesIn, y); //zzz=E_em/E_obs
      double freq_local = freq_obs*zzz;  

      // thermal synchrotron
      double nth0=3e7;
      double zc=r*cos(theta);
      double rc=r*sin(theta);

      double nth=nth0*exp(-zc*zc/2./rc/rc)*pow(r,-1.1);
      double Te=1.7e11*pow(r,-0.84); 
      double b=sqrt(8.*PI*0.1*nth*C_mp*C_c*C_c/6./r);

      double vb=C_e*b/2./PI/C_me/C_c;
      double theta_E= C_kB*Te/C_me/C_c/C_c;
      double v=freq_local;
      double x=2.*v/3./vb/theta_E/theta_E;

      double K_value=cpu_K2(theta_E);

      double comp1=4.*PI*nth*C_e*C_e*v/sqrt(3.)/K_value/C_c;
      double comp2=4.0505/pow(x,(1./6.))*(1.+0.4/pow(x,0.25)+0.5316/sqrt(x))*exp(-1.8899*pow(x,1./3.));
      double j_nu=comp1*comp2;
      double B_nu=2.0*v*v*v*C_h/C_c/C_c/(exp(C_h*v/C_kB/Te)-1.0);

      // integrate intensity along ray
      ds    =  hdid; cpu_transfer_steps++; if(htry != hdid) cpu_transfer_rejected_steps++;

      dtau  =  dtau + ds*C_sgrA_mbh*C_rgeo*j_nu/B_nu*zzz;  
      dI    =  dI + ds*C_sgrA_mbh*C_rgeo*j_nu/freq_local/freq_local/freq_local*exp(-dtau)*zzz;  
    }
    htry = hnext;
  }
}

}

namespace d14 {
static double cpu_event[64][3]={};
// Deterministic host-only clone of original CUDA source; qualifiers/index mapping changed, diagnostics added without changing equations.
static int cpu_x,cpu_y; static long long cpu_accepted_steps=0,cpu_shrunk_steps=0,cpu_transfer_steps=0,cpu_transfer_rejected_steps=0; static double cpu_initial_null_max=0;
/*
  Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
  Odyssey  version 1.0   (released  2015)
  This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
  for General Relativistic Radiative Transfer (GRRT), following the 
  ray-tracing algorithm presented in 
  Fuerst, S. V., & Wu, K. 2007, A&A, 474, 55, 
  and the radiative transfer formulation described in 
  Younsi, Z., Wu, K., & Fuerst, S. V. 2012, A&A, 545, A13

  Odyssey is distributed freely under the GNU general public license. 
  You can redistribute it and/or modify it under the terms of the License
  http://www.gnu.org/licenses/gpl.txt
  The current distribution website is:
  https://github.com/hungyipu/Odyssey/ 
 */


 static void cpu_geodesic(double* Variables, double* VariablesIn, double *y, double *dydx)
{
  double r = y[0];
  double theta = y[1];
  double pr = y[4];
  double ptheta = y[5];

  double r2  = r * r;
  double twor = 2.0 * r;

  double sintheta = sin(theta);
  double costheta = cos(theta);
  double cos2     = costheta * costheta;
  double sin2     = sintheta * sintheta;
  double sigma  = r2 + a2 * cos2;
  double delta  = r2 - twor + a2;
  double sd    = sigma * delta;
  double siginv  = 1.0 / sigma;
  double bot    = 1.0 / sd;


  //avoid problems near the axis
  if (sintheta < 1e-8)
  {
    sintheta = 1e-8;
    sin2 = 1e-16;
  }

  dydx[0] = -pr * delta * siginv;  
  dydx[1] = -ptheta * siginv;
  dydx[2] = -(twor * A + (sigma - twor) * L / sin2) * bot;
  dydx[3] = -(1.0 + (twor * (r2 + a2) - twor * A * L) * bot);
  dydx[4] = -(((r - 1.0) * (-kappa) + twor * (r2 + a2) - 2.0 * A * L) * bot - 2.0 * pr * pr * (r - 1.0) * siginv);
  dydx[5] = -sintheta * costheta*(L * L / (sin2 * sin2) - a2) * siginv;
}

 static void cpu_rkstep(double* Variables, double* VariablesIn,double *y,
                              double *dydx, double h, double *yout, double *yerr)
{
  int i;
  double ak[N];
  double ytemp1[N], ytemp2[N], ytemp3[N], ytemp4[N], ytemp5[N];
  double hdydx;
  double yi, yt;

  for (i = 0; i < N; i++)
  {
    hdydx     = h * dydx[i];
    yi        = y[i];
    ytemp1[i] = yi + 0.2 * hdydx;
    ytemp2[i] = yi + (3.0/40.0) * hdydx;
    ytemp3[i] = yi + 0.3 * hdydx;
    ytemp4[i] = yi -(11.0/54.0) * hdydx;
    ytemp5[i] = yi + (1631.0/55296.0) * hdydx;
    yout[i]   = yi + (37.0/378.0) * hdydx;
    yerr[i]   = ((37.0/378.0)-(2825.0/27648.0)) * hdydx;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp1, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp2[i] += (9.0/40.0) * yt;
    ytemp3[i] -= 0.9 * yt;
    ytemp4[i] += 2.5 * yt;
    ytemp5[i] += (175.0/512.0) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp2, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp3[i] += 1.2 * yt;
    ytemp4[i] -= (70.0/27.0) * yt;
    ytemp5[i] += (575.0/13824.0) * yt;
    yout[i]   += (250.0/621.0) * yt;
    yerr[i]   += ((250.0/621.0)-(18575.0/48384.0)) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp3, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp4[i] += (35.0/27.0) * yt;
    ytemp5[i] += (44275.0/110592.0) * yt;
    yout[i]   += (125.0/594.0) * yt;
    yerr[i]   += ((125.0/594.0)-(13525.0/55296.0)) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp4, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp5[i] += (253.0/4096.0) * yt;
    yerr[i]   -= (277.0/14336.0) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp5, ak);

  for (i = 0; i < N; i++)
  {
    yt       = h * ak[i];
    yout[i] += (512.0/1771.0) * yt;
    yerr[i] += ((512.0/1771.0)-0.25) * yt;
  }
}

 static double cpu_rk5(double* Variables, double* VariablesIn, double *y, double *dydx, 
                             double htry, double escal, double *yscal, double *hdid)
{
  int i;
  double hnext;
  double errmax, h = htry, htemp;
  double yerr[N], ytemp[N];

  while (1)
  {
    // find adaptive step size
    cpu_rkstep(Variables, VariablesIn, y, dydx, h, ytemp, yerr);

    errmax = 0.0;
    for (i = 0; i < N; i++)
    {
      double temp = fabs(yerr[i]/yscal[i]);
      if (temp > errmax) errmax = temp;
    }

    errmax *= escal;
    if (errmax <= 1.0) break;

    htemp = 0.9 * h / sqrt(sqrt(errmax));

    h *= 0.1;

    if (h >= 0.0)
    {
      if (htemp > h) h = htemp;
    }
    else
    {
      if (htemp < h) h = htemp;
    }
  }

  if (errmax > 1.89e-4)
  {
    hnext = 0.9 * h * pow(errmax, -0.2);
  }
  else
  {
    hnext = 5.0 * h;
  }

  *hdid = h; cpu_accepted_steps++; if(h != htry) cpu_shrunk_steps++;

  memcpy(y, ytemp, N * sizeof(double));

  return hnext;
}

 static void cpu_initial(double* Variables, double* VariablesIn, double *y0, double *ydot0)
{
  double alpha = grid_x;
  double beta  = grid_y;

  //see Eq[18-19], with phi_obs=0, x= alpha, y=beta, z=0
  double x     = sqrt(r0*r0+a2)*sin(theta0)-beta*cos(theta0);
  double y     = alpha;
  double z     = r0*cos(theta0)+beta*sin(theta0);
  double w     = x*x+y*y+z*z-a2;

  //see Eq[20-22]
  y0[0] = sqrt((w+sqrt(w*w+(2.*A*z)*(2.*A*z)))/2.);   
  y0[1] = acos(z/y0[0]);                              
  y0[2] = atan2(y,x);                                
  y0[3] = 0;
  double r = y0[0];
  double theta = y0[1];
  double phi=y0[2];

  double sigma = r*r+(A*cos(theta))*(A*cos(theta));
  double u     = sqrt(a2+r*r);
  double v     = -sin(theta0)*cos(phi);
  double zdot  = -1.;

  //see Eq[24-26]
  double rdot0 = zdot*(-u*u*cos(theta0)*cos(theta)+r*u*v*sin(theta))/sigma;         
  double thetadot0 = zdot*(cos(theta0)*r*sin(theta)+u*v*cos(theta))/sigma;          
  double phidot0 = zdot*sin(theta0)*sin(phi)/(u*sin(theta));                         

  ydot0[0] = rdot0;
  ydot0[1] = thetadot0;
  ydot0[2] = phidot0;

  double sintheta=sin(theta);
  double sin2 = sintheta*sintheta;

  double r2 = r * r;
  double delta = r2 - 2.0 * r + a2;
  double s1 = sigma - 2.0 * r;

  y0[4] = rdot0*sigma/delta;
  y0[5] = thetadot0*sigma;

  // Eq.7
  double energy2 = s1*(rdot0*rdot0/delta+thetadot0*thetadot0)
    + delta*sin2*phidot0*phidot0;

  double energy = sqrt(energy2);

  // rescaled by energy
  y0[4] = y0[4]/energy;
  y0[5] = y0[5]/energy;

  // Eq.8 
  L = ((sigma*delta*phidot0-2.0*A*r*energy)*sin2/s1)/energy;

  // Eq below Eq.15
  kappa = y0[5]*y0[5]+a2*sin2+L*L/sin2; if(A==0){double H=-1/(1-2/r)+(1-2/r)*y0[4]*y0[4]+y0[5]*y0[5]/r2+L*L/(r2*sin2);cpu_initial_null_max=fmax(cpu_initial_null_max,fabs(H));}
}

 static float cpu_ISCO(double* VariablesIn)
{
  double z1       = 1 + pow(1 - A * A, 1 / 3.0) * pow(1 + A, 1 / 3.0) + pow(1 - A, 1 / 3.0);
  double z2       = sqrt(3 * A * A + z1 * z1);
  return 3. + z2 - sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

//
// functions for radiative transfer                                                  
//

#define Te_min    0.1
#define Te_max    100.
#define Te_grids  50.

static   double cpu_K2_tab[] = {
  -10.747001, //Te=0.1
  -9.362569,
  -8.141373,
  -7.061568,
  -6.104060,
  -5.252153,
  -4.491244,
  -3.808555,
  -3.192909,
  -2.634534,
  -2.124893,
  -1.656543,
  -1.223007,
  -0.818668,
  -0.438676,
  -0.078863,
  +0.264332,
  +0.593930,
  +0.912476,
  +1.222098,
  +1.524560,
  +1.821311,
  +2.113537,
  +2.402193,
  +2.688050,
  +2.971721,
  +3.253692,
  +3.534347,
  +3.813984,
  +4.092839,
  +4.371092,
  +4.648884,
  +4.926323,
  +5.203493,
  +5.480457,
  +5.757264,
  +6.033952,
  +6.310550,
  +6.587078,
  +6.863554,
  +7.139990,
  +7.416395,
  +7.692778,
  +7.969143,
  +8.245495,
  +8.521837,
  +8.798171,
  +9.074500,
  +9.350824,
  +9.627144  //Te=87.09
};

 static double cpu_K2_find(double Te)
{
  double d = Te_grids*(log(Te / Te_min)/ log(Te_max / Te_min));
  int    i = floor(d);

  return (1 - (double)(d-i)) * cpu_K2_tab[i] + (double)(d-i) * cpu_K2_tab[i+1];
}

 static double cpu_K2(double Te)
{
  double tab_K2;
  //avoid the boundary effect when T~T_max
  if (Te>85.){ 
    tab_K2=2.*Te*Te;
    return tab_K2;
  }

  if (Te<Te_min){ 
    //set a dummy value
    return exp(-11.);
  }

  tab_K2= cpu_K2_find(Te);
  return exp(tab_K2);
}

 static double cpu_Jansky_Correction(double* VariablesIn,double ima_width)
{
  double distance=C_sgrA_d*C_pc;
  double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
  double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //radian^2 (steradians) per pixel
  return pix_str/C_Jansky;
}

 static double cpu_Luminosity_Correction(double* VariablesIn,double ima_width)
{
  double distance=C_sgrA_d*C_pc;
  double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
  double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //radian^2 (steradians) per pixel
  return pix_str*distance*distance*4.*PI*freq_obs;
}

 double cpu_task1fun_GetZ(double* Variables, double* VariablesIn, double *y)
{
  double r1 = y[0];
  double E_local = -(r1 * r1 + A * sqrt(r1)) / (r1 * sqrt(r1 * r1 - 3. * r1 + 2. * A * sqrt(r1))) + 
                  L / sqrt(r1) / sqrt(r1 * r1 - 3. * r1  +2. * A * sqrt(r1));
  double E_inf = -1.0;      
  return E_local / E_inf; 
}

 void cpu_task1(double*__restrict__ ResultsPixel,
                      double*__restrict__ VariablesIn,
                      int GridIdxX, int GridIdxY)
{
  // to check whether the photon is inside image plane
  if(cpu_x >= SIZE || cpu_y >= SIZE) return;  

  double Variables[VarNUM];

  r0       = 1000.0;                 
  theta0   = (PI/180.0) * INCLINATION;     
  a2       = A * A;                
  Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;      
  Rmstable = cpu_ISCO(VariablesIn);                

  double htry = 0.5, escal = 1e14, hdid = 0.0, hnext = 0.0;
  double y[N], dydx[N], yscal[N], ylaststep[N];
  double Rdisk   = 50.;
  double ima_width = 55.;
  double s1  = ima_width;                      
  double s2  = 2.*ima_width/((int)SIZE+1.);   

  grid_x = -s1 + s2*(cpu_x+1.);
  grid_y = -s1 + s2*(cpu_y+1.);

  cpu_initial(Variables, VariablesIn, y, dydx);

  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+0] = grid_x;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+1] = grid_y;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 0;

  while (1)
  {
    for(int i = 0; i < N; i++) ylaststep[i] = y[i];

    cpu_geodesic(Variables, VariablesIn, y, dydx);

    for (int i = 0; i < N; i++) yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

    //fifth-order Runge-Kutta method
    hnext = cpu_rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

    // hit the disk, compute redshift
    if( y[0] < Rdisk && y[0] > Rmstable && (ylaststep[1] - PI/2.) * (y[1] - PI/2.) < 0. )
    {    
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 1./cpu_task1fun_GetZ(Variables, VariablesIn, y);
      double fraction=(PI/2.-ylaststep[1])/(y[1]-ylaststep[1]);
      double plane[N];for(int q=0;q<N;q++)plane[q]=ylaststep[q]+fraction*(y[q]-ylaststep[q]);
      int pix=IMAGE_SIZE*cpu_y+cpu_x;
      cpu_event[pix][0]=ResultsPixel[3*pix+2];
      cpu_event[pix][1]=1./cpu_task1fun_GetZ(Variables,VariablesIn,plane);
      cpu_event[pix][2]=fabs(1./cpu_task1fun_GetZ(Variables,VariablesIn,ylaststep)-cpu_event[pix][0]);

      break;
    }

    // Inside the event horizon radius or escape to infinity
    if ((y[0] > r0) && (dydx[0]>0)) break;

    if (y[0] < Rhor) break;

    htry = hnext;
  }
}

 double cpu_task2fun_GetZ(double* Variables, double* VariablesIn, double *y)
{
  double ut,uphi,ur,E_local;
  double E_inf= -1.0;   
  double r=y[0];
  double theta=y[1];
  double pr=y[4];

  double r2 = r*r;
  double twor = 2.0*r;
  double sintheta, costheta;
  sintheta=sin(theta);
  costheta=cos(theta);
  double cos2 = costheta*costheta;
  double sin2 = sintheta*sintheta;

  double sigma = r2+a2*cos2;
  double delta = r2-twor+a2;
  double ssig=(r2+a2)*(r2+a2)-a2*delta*sin2; 

  //======define (covariant) metric component    
  double gtt=-(1.-2.*r/sigma);
  double gtph=-2.*A*r*sin2/sigma;
  double grr=sigma/delta;
  double gphph=ssig*sin2/sigma;    


  //==========Keplerian flow: outside cpu_ISCO
  double ut_k  =(r*r+A*sqrt(r))/(r*sqrt(r*r-3.*r+2.*A*sqrt(r)));
  double ur_k  =0.;
  double uphi_k  =1./(sqrt(r)*sqrt(r*r-3.*r+2.*A*sqrt(r)));

  //==========Keplerian flow: inside cpu_ISCO    
  if( r<Rmstable)
  {

    double delta = r*r-2.*r+a2;
    double lambda=(Rmstable*Rmstable-2.*A*sqrt(Rmstable)+a2)/(sqrt(Rmstable*Rmstable*Rmstable)-2.*sqrt(Rmstable)+A);
    double gamma=sqrt(1-2./3./Rmstable);
    double h=(2.*r-A*lambda)/delta;

    ut_k=gamma*(1.+2/r*(1.+h));
    ur_k=-sqrt(2./3./Rmstable)*sqrt(pow((Rmstable/r-1.),3.));
    uphi_k=gamma/r/r*(lambda+A*h);
  }  

  //normalize the four-velocity
  ut    = ut_k;
  uphi  = uphi_k;
  ur    = ur_k;  
  double omega = uphi/ut;
  double k0    = -(gtt + omega*omega*gphph + 2.*omega*gtph);
  ut = sqrt(((1. + grr*ur*ur) / k0));
  uphi = omega*ut;  

  // compute redshift
  E_local=-ut+L*uphi+pr*ur;
  return E_local/E_inf; 
}

 void cpu_task2(double*__restrict__ ResultsPixel,
                      double*__restrict__ VariablesIn,
                      int GridIdxX, int GridIdxY)
{
  if(cpu_x >= SIZE || cpu_y >= SIZE) return;  

  double Variables[VarNUM];
  r0       = 1000.0;                 
  theta0   = (PI/180.0) * INCLINATION;     
  a2       = A * A;                
  Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;   
  Rmstable = cpu_ISCO(VariablesIn);  

  double htry = 0.5, escal = 1e14, hdid = 0.0, hnext = 0.0;
  double y[N], dydx[N], yscal[N];

  double Rdisk   = 500.;
  double ima_width = 10.;
  double s1  = ima_width;                      
  double s2  = 2.*ima_width/((int)SIZE+1.);   
  double Jy_corr=cpu_Jansky_Correction(VariablesIn,ima_width);
  double L_corr=cpu_Luminosity_Correction(VariablesIn,ima_width);

  grid_x = -s1 + s2*(cpu_x+1.);
  grid_y = -s1 + s2*(cpu_y+1.);

  cpu_initial(Variables, VariablesIn, y, dydx);

  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+0] = grid_x;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+1] = grid_y;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 0;

  double ds=0.;  
  double dtau=0.;
  double dI=0.;

  while (1)
  {
    cpu_geodesic(Variables, VariablesIn, y, dydx);

    for (int i = 0; i < N; i++)
      yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

    hnext = cpu_rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

    if ((y[0] > r0) && (dydx[0]>0)){
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = dI*freq_obs*freq_obs*freq_obs*L_corr; 
      break;
    }

    if (y[0] < Rhor){
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = dI*freq_obs*freq_obs*freq_obs*L_corr; 
      break;
    }

    double r=y[0];
    double theta=y[1];

    if(y[0]<Rdisk){

      double zzz        = cpu_task2fun_GetZ(Variables, VariablesIn, y); //zzz=E_em/E_obs
      double freq_local = freq_obs*zzz;  

      // thermal synchrotron
      double nth0=3e7;
      double zc=r*cos(theta);
      double rc=r*sin(theta);

      double nth=nth0*exp(-zc*zc/2./rc/rc)*pow(r,-1.1);
      double Te=1.7e11*pow(r,-0.84); 
      double b=sqrt(8.*PI*0.1*nth*C_mp*C_c*C_c/6./r);

      double vb=C_e*b/2./PI/C_me/C_c;
      double theta_E= C_kB*Te/C_me/C_c/C_c;
      double v=freq_local;
      double x=2.*v/3./vb/theta_E/theta_E;

      double K_value=cpu_K2(theta_E);

      double comp1=4.*PI*nth*C_e*C_e*v/sqrt(3.)/K_value/C_c;
      double comp2=4.0505/pow(x,(1./6.))*(1.+0.4/pow(x,0.25)+0.5316/sqrt(x))*exp(-1.8899*pow(x,1./3.));
      double j_nu=comp1*comp2;
      double B_nu=2.0*v*v*v*C_h/C_c/C_c/(exp(C_h*v/C_kB/Te)-1.0);

      // integrate intensity along ray
      ds    =  hdid; cpu_transfer_steps++; if(htry != hdid) cpu_transfer_rejected_steps++;

      dtau  =  dtau + ds*C_sgrA_mbh*C_rgeo*j_nu/B_nu*zzz;  
      dI    =  dI + ds*C_sgrA_mbh*C_rgeo*j_nu/freq_local/freq_local/freq_local*exp(-dtau)*zzz;  
    }
    htry = hnext;
  }
}

}

namespace d15 {
static double cpu_event[64][3]={};
// Deterministic host-only clone of original CUDA source; qualifiers/index mapping changed, diagnostics added without changing equations.
static int cpu_x,cpu_y; static long long cpu_accepted_steps=0,cpu_shrunk_steps=0,cpu_transfer_steps=0,cpu_transfer_rejected_steps=0; static double cpu_initial_null_max=0;
/*
  Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
  Odyssey  version 1.0   (released  2015)
  This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
  for General Relativistic Radiative Transfer (GRRT), following the 
  ray-tracing algorithm presented in 
  Fuerst, S. V., & Wu, K. 2007, A&A, 474, 55, 
  and the radiative transfer formulation described in 
  Younsi, Z., Wu, K., & Fuerst, S. V. 2012, A&A, 545, A13

  Odyssey is distributed freely under the GNU general public license. 
  You can redistribute it and/or modify it under the terms of the License
  http://www.gnu.org/licenses/gpl.txt
  The current distribution website is:
  https://github.com/hungyipu/Odyssey/ 
 */


 static void cpu_geodesic(double* Variables, double* VariablesIn, double *y, double *dydx)
{
  double r = y[0];
  double theta = y[1];
  double pr = y[4];
  double ptheta = y[5];

  double r2  = r * r;
  double twor = 2.0 * r;

  double sintheta = sin(theta);
  double costheta = cos(theta);
  double cos2     = costheta * costheta;
  double sin2     = sintheta * sintheta;
  double sigma  = r2 + a2 * cos2;
  double delta  = r2 - twor + a2;
  double sd    = sigma * delta;
  double siginv  = 1.0 / sigma;
  double bot    = 1.0 / sd;


  //avoid problems near the axis
  if (sintheta < 1e-8)
  {
    sintheta = 1e-8;
    sin2 = 1e-16;
  }

  dydx[0] = -pr * delta * siginv;  
  dydx[1] = -ptheta * siginv;
  dydx[2] = -(twor * A + (sigma - twor) * L / sin2) * bot;
  dydx[3] = -(1.0 + (twor * (r2 + a2) - twor * A * L) * bot);
  dydx[4] = -(((r - 1.0) * (-kappa) + twor * (r2 + a2) - 2.0 * A * L) * bot - 2.0 * pr * pr * (r - 1.0) * siginv);
  dydx[5] = -sintheta * costheta*(L * L / (sin2 * sin2) - a2) * siginv;
}

 static void cpu_rkstep(double* Variables, double* VariablesIn,double *y,
                              double *dydx, double h, double *yout, double *yerr)
{
  int i;
  double ak[N];
  double ytemp1[N], ytemp2[N], ytemp3[N], ytemp4[N], ytemp5[N];
  double hdydx;
  double yi, yt;

  for (i = 0; i < N; i++)
  {
    hdydx     = h * dydx[i];
    yi        = y[i];
    ytemp1[i] = yi + 0.2 * hdydx;
    ytemp2[i] = yi + (3.0/40.0) * hdydx;
    ytemp3[i] = yi + 0.3 * hdydx;
    ytemp4[i] = yi -(11.0/54.0) * hdydx;
    ytemp5[i] = yi + (1631.0/55296.0) * hdydx;
    yout[i]   = yi + (37.0/378.0) * hdydx;
    yerr[i]   = ((37.0/378.0)-(2825.0/27648.0)) * hdydx;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp1, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp2[i] += (9.0/40.0) * yt;
    ytemp3[i] -= 0.9 * yt;
    ytemp4[i] += 2.5 * yt;
    ytemp5[i] += (175.0/512.0) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp2, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp3[i] += 1.2 * yt;
    ytemp4[i] -= (70.0/27.0) * yt;
    ytemp5[i] += (575.0/13824.0) * yt;
    yout[i]   += (250.0/621.0) * yt;
    yerr[i]   += ((250.0/621.0)-(18575.0/48384.0)) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp3, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp4[i] += (35.0/27.0) * yt;
    ytemp5[i] += (44275.0/110592.0) * yt;
    yout[i]   += (125.0/594.0) * yt;
    yerr[i]   += ((125.0/594.0)-(13525.0/55296.0)) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp4, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp5[i] += (253.0/4096.0) * yt;
    yerr[i]   -= (277.0/14336.0) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp5, ak);

  for (i = 0; i < N; i++)
  {
    yt       = h * ak[i];
    yout[i] += (512.0/1771.0) * yt;
    yerr[i] += ((512.0/1771.0)-0.25) * yt;
  }
}

 static double cpu_rk5(double* Variables, double* VariablesIn, double *y, double *dydx, 
                             double htry, double escal, double *yscal, double *hdid)
{
  int i;
  double hnext;
  double errmax, h = htry, htemp;
  double yerr[N], ytemp[N];

  while (1)
  {
    // find adaptive step size
    cpu_rkstep(Variables, VariablesIn, y, dydx, h, ytemp, yerr);

    errmax = 0.0;
    for (i = 0; i < N; i++)
    {
      double temp = fabs(yerr[i]/yscal[i]);
      if (temp > errmax) errmax = temp;
    }

    errmax *= escal;
    if (errmax <= 1.0) break;

    htemp = 0.9 * h / sqrt(sqrt(errmax));

    h *= 0.1;

    if (h >= 0.0)
    {
      if (htemp > h) h = htemp;
    }
    else
    {
      if (htemp < h) h = htemp;
    }
  }

  if (errmax > 1.89e-4)
  {
    hnext = 0.9 * h * pow(errmax, -0.2);
  }
  else
  {
    hnext = 5.0 * h;
  }

  *hdid = h; cpu_accepted_steps++; if(h != htry) cpu_shrunk_steps++;

  memcpy(y, ytemp, N * sizeof(double));

  return hnext;
}

 static void cpu_initial(double* Variables, double* VariablesIn, double *y0, double *ydot0)
{
  double alpha = grid_x;
  double beta  = grid_y;

  //see Eq[18-19], with phi_obs=0, x= alpha, y=beta, z=0
  double x     = sqrt(r0*r0+a2)*sin(theta0)-beta*cos(theta0);
  double y     = alpha;
  double z     = r0*cos(theta0)+beta*sin(theta0);
  double w     = x*x+y*y+z*z-a2;

  //see Eq[20-22]
  y0[0] = sqrt((w+sqrt(w*w+(2.*A*z)*(2.*A*z)))/2.);   
  y0[1] = acos(z/y0[0]);                              
  y0[2] = atan2(y,x);                                
  y0[3] = 0;
  double r = y0[0];
  double theta = y0[1];
  double phi=y0[2];

  double sigma = r*r+(A*cos(theta))*(A*cos(theta));
  double u     = sqrt(a2+r*r);
  double v     = -sin(theta0)*cos(phi);
  double zdot  = -1.;

  //see Eq[24-26]
  double rdot0 = zdot*(-u*u*cos(theta0)*cos(theta)+r*u*v*sin(theta))/sigma;         
  double thetadot0 = zdot*(cos(theta0)*r*sin(theta)+u*v*cos(theta))/sigma;          
  double phidot0 = zdot*sin(theta0)*sin(phi)/(u*sin(theta));                         

  ydot0[0] = rdot0;
  ydot0[1] = thetadot0;
  ydot0[2] = phidot0;

  double sintheta=sin(theta);
  double sin2 = sintheta*sintheta;

  double r2 = r * r;
  double delta = r2 - 2.0 * r + a2;
  double s1 = sigma - 2.0 * r;

  y0[4] = rdot0*sigma/delta;
  y0[5] = thetadot0*sigma;

  // Eq.7
  double energy2 = s1*(rdot0*rdot0/delta+thetadot0*thetadot0)
    + delta*sin2*phidot0*phidot0;

  double energy = sqrt(energy2);

  // rescaled by energy
  y0[4] = y0[4]/energy;
  y0[5] = y0[5]/energy;

  // Eq.8 
  L = ((sigma*delta*phidot0-2.0*A*r*energy)*sin2/s1)/energy;

  // Eq below Eq.15
  kappa = y0[5]*y0[5]+a2*sin2+L*L/sin2; if(A==0){double H=-1/(1-2/r)+(1-2/r)*y0[4]*y0[4]+y0[5]*y0[5]/r2+L*L/(r2*sin2);cpu_initial_null_max=fmax(cpu_initial_null_max,fabs(H));}
}

 static float cpu_ISCO(double* VariablesIn)
{
  double z1       = 1 + pow(1 - A * A, 1 / 3.0) * pow(1 + A, 1 / 3.0) + pow(1 - A, 1 / 3.0);
  double z2       = sqrt(3 * A * A + z1 * z1);
  return 3. + z2 - sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

//
// functions for radiative transfer                                                  
//

#define Te_min    0.1
#define Te_max    100.
#define Te_grids  50.

static   double cpu_K2_tab[] = {
  -10.747001, //Te=0.1
  -9.362569,
  -8.141373,
  -7.061568,
  -6.104060,
  -5.252153,
  -4.491244,
  -3.808555,
  -3.192909,
  -2.634534,
  -2.124893,
  -1.656543,
  -1.223007,
  -0.818668,
  -0.438676,
  -0.078863,
  +0.264332,
  +0.593930,
  +0.912476,
  +1.222098,
  +1.524560,
  +1.821311,
  +2.113537,
  +2.402193,
  +2.688050,
  +2.971721,
  +3.253692,
  +3.534347,
  +3.813984,
  +4.092839,
  +4.371092,
  +4.648884,
  +4.926323,
  +5.203493,
  +5.480457,
  +5.757264,
  +6.033952,
  +6.310550,
  +6.587078,
  +6.863554,
  +7.139990,
  +7.416395,
  +7.692778,
  +7.969143,
  +8.245495,
  +8.521837,
  +8.798171,
  +9.074500,
  +9.350824,
  +9.627144  //Te=87.09
};

 static double cpu_K2_find(double Te)
{
  double d = Te_grids*(log(Te / Te_min)/ log(Te_max / Te_min));
  int    i = floor(d);

  return (1 - (double)(d-i)) * cpu_K2_tab[i] + (double)(d-i) * cpu_K2_tab[i+1];
}

 static double cpu_K2(double Te)
{
  double tab_K2;
  //avoid the boundary effect when T~T_max
  if (Te>85.){ 
    tab_K2=2.*Te*Te;
    return tab_K2;
  }

  if (Te<Te_min){ 
    //set a dummy value
    return exp(-11.);
  }

  tab_K2= cpu_K2_find(Te);
  return exp(tab_K2);
}

 static double cpu_Jansky_Correction(double* VariablesIn,double ima_width)
{
  double distance=C_sgrA_d*C_pc;
  double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
  double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //radian^2 (steradians) per pixel
  return pix_str/C_Jansky;
}

 static double cpu_Luminosity_Correction(double* VariablesIn,double ima_width)
{
  double distance=C_sgrA_d*C_pc;
  double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
  double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //radian^2 (steradians) per pixel
  return pix_str*distance*distance*4.*PI*freq_obs;
}

 double cpu_task1fun_GetZ(double* Variables, double* VariablesIn, double *y)
{
  double r1 = y[0];
  double E_local = -(r1 * r1 + A * sqrt(r1)) / (r1 * sqrt(r1 * r1 - 3. * r1 + 2. * A * sqrt(r1))) + 
                  L / sqrt(r1) / sqrt(r1 * r1 - 3. * r1  +2. * A * sqrt(r1));
  double E_inf = -1.0;      
  return E_local / E_inf; 
}

 void cpu_task1(double*__restrict__ ResultsPixel,
                      double*__restrict__ VariablesIn,
                      int GridIdxX, int GridIdxY)
{
  // to check whether the photon is inside image plane
  if(cpu_x >= SIZE || cpu_y >= SIZE) return;  

  double Variables[VarNUM];

  r0       = 1000.0;                 
  theta0   = (PI/180.0) * INCLINATION;     
  a2       = A * A;                
  Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;      
  Rmstable = cpu_ISCO(VariablesIn);                

  double htry = 0.5, escal = 1e15, hdid = 0.0, hnext = 0.0;
  double y[N], dydx[N], yscal[N], ylaststep[N];
  double Rdisk   = 50.;
  double ima_width = 55.;
  double s1  = ima_width;                      
  double s2  = 2.*ima_width/((int)SIZE+1.);   

  grid_x = -s1 + s2*(cpu_x+1.);
  grid_y = -s1 + s2*(cpu_y+1.);

  cpu_initial(Variables, VariablesIn, y, dydx);

  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+0] = grid_x;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+1] = grid_y;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 0;

  while (1)
  {
    for(int i = 0; i < N; i++) ylaststep[i] = y[i];

    cpu_geodesic(Variables, VariablesIn, y, dydx);

    for (int i = 0; i < N; i++) yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

    //fifth-order Runge-Kutta method
    hnext = cpu_rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

    // hit the disk, compute redshift
    if( y[0] < Rdisk && y[0] > Rmstable && (ylaststep[1] - PI/2.) * (y[1] - PI/2.) < 0. )
    {    
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 1./cpu_task1fun_GetZ(Variables, VariablesIn, y);
      double fraction=(PI/2.-ylaststep[1])/(y[1]-ylaststep[1]);
      double plane[N];for(int q=0;q<N;q++)plane[q]=ylaststep[q]+fraction*(y[q]-ylaststep[q]);
      int pix=IMAGE_SIZE*cpu_y+cpu_x;
      cpu_event[pix][0]=ResultsPixel[3*pix+2];
      cpu_event[pix][1]=1./cpu_task1fun_GetZ(Variables,VariablesIn,plane);
      cpu_event[pix][2]=fabs(1./cpu_task1fun_GetZ(Variables,VariablesIn,ylaststep)-cpu_event[pix][0]);

      break;
    }

    // Inside the event horizon radius or escape to infinity
    if ((y[0] > r0) && (dydx[0]>0)) break;

    if (y[0] < Rhor) break;

    htry = hnext;
  }
}

 double cpu_task2fun_GetZ(double* Variables, double* VariablesIn, double *y)
{
  double ut,uphi,ur,E_local;
  double E_inf= -1.0;   
  double r=y[0];
  double theta=y[1];
  double pr=y[4];

  double r2 = r*r;
  double twor = 2.0*r;
  double sintheta, costheta;
  sintheta=sin(theta);
  costheta=cos(theta);
  double cos2 = costheta*costheta;
  double sin2 = sintheta*sintheta;

  double sigma = r2+a2*cos2;
  double delta = r2-twor+a2;
  double ssig=(r2+a2)*(r2+a2)-a2*delta*sin2; 

  //======define (covariant) metric component    
  double gtt=-(1.-2.*r/sigma);
  double gtph=-2.*A*r*sin2/sigma;
  double grr=sigma/delta;
  double gphph=ssig*sin2/sigma;    


  //==========Keplerian flow: outside cpu_ISCO
  double ut_k  =(r*r+A*sqrt(r))/(r*sqrt(r*r-3.*r+2.*A*sqrt(r)));
  double ur_k  =0.;
  double uphi_k  =1./(sqrt(r)*sqrt(r*r-3.*r+2.*A*sqrt(r)));

  //==========Keplerian flow: inside cpu_ISCO    
  if( r<Rmstable)
  {

    double delta = r*r-2.*r+a2;
    double lambda=(Rmstable*Rmstable-2.*A*sqrt(Rmstable)+a2)/(sqrt(Rmstable*Rmstable*Rmstable)-2.*sqrt(Rmstable)+A);
    double gamma=sqrt(1-2./3./Rmstable);
    double h=(2.*r-A*lambda)/delta;

    ut_k=gamma*(1.+2/r*(1.+h));
    ur_k=-sqrt(2./3./Rmstable)*sqrt(pow((Rmstable/r-1.),3.));
    uphi_k=gamma/r/r*(lambda+A*h);
  }  

  //normalize the four-velocity
  ut    = ut_k;
  uphi  = uphi_k;
  ur    = ur_k;  
  double omega = uphi/ut;
  double k0    = -(gtt + omega*omega*gphph + 2.*omega*gtph);
  ut = sqrt(((1. + grr*ur*ur) / k0));
  uphi = omega*ut;  

  // compute redshift
  E_local=-ut+L*uphi+pr*ur;
  return E_local/E_inf; 
}

 void cpu_task2(double*__restrict__ ResultsPixel,
                      double*__restrict__ VariablesIn,
                      int GridIdxX, int GridIdxY)
{
  if(cpu_x >= SIZE || cpu_y >= SIZE) return;  

  double Variables[VarNUM];
  r0       = 1000.0;                 
  theta0   = (PI/180.0) * INCLINATION;     
  a2       = A * A;                
  Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;   
  Rmstable = cpu_ISCO(VariablesIn);  

  double htry = 0.5, escal = 1e15, hdid = 0.0, hnext = 0.0;
  double y[N], dydx[N], yscal[N];

  double Rdisk   = 500.;
  double ima_width = 10.;
  double s1  = ima_width;                      
  double s2  = 2.*ima_width/((int)SIZE+1.);   
  double Jy_corr=cpu_Jansky_Correction(VariablesIn,ima_width);
  double L_corr=cpu_Luminosity_Correction(VariablesIn,ima_width);

  grid_x = -s1 + s2*(cpu_x+1.);
  grid_y = -s1 + s2*(cpu_y+1.);

  cpu_initial(Variables, VariablesIn, y, dydx);

  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+0] = grid_x;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+1] = grid_y;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 0;

  double ds=0.;  
  double dtau=0.;
  double dI=0.;

  while (1)
  {
    cpu_geodesic(Variables, VariablesIn, y, dydx);

    for (int i = 0; i < N; i++)
      yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

    hnext = cpu_rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

    if ((y[0] > r0) && (dydx[0]>0)){
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = dI*freq_obs*freq_obs*freq_obs*L_corr; 
      break;
    }

    if (y[0] < Rhor){
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = dI*freq_obs*freq_obs*freq_obs*L_corr; 
      break;
    }

    double r=y[0];
    double theta=y[1];

    if(y[0]<Rdisk){

      double zzz        = cpu_task2fun_GetZ(Variables, VariablesIn, y); //zzz=E_em/E_obs
      double freq_local = freq_obs*zzz;  

      // thermal synchrotron
      double nth0=3e7;
      double zc=r*cos(theta);
      double rc=r*sin(theta);

      double nth=nth0*exp(-zc*zc/2./rc/rc)*pow(r,-1.1);
      double Te=1.7e11*pow(r,-0.84); 
      double b=sqrt(8.*PI*0.1*nth*C_mp*C_c*C_c/6./r);

      double vb=C_e*b/2./PI/C_me/C_c;
      double theta_E= C_kB*Te/C_me/C_c/C_c;
      double v=freq_local;
      double x=2.*v/3./vb/theta_E/theta_E;

      double K_value=cpu_K2(theta_E);

      double comp1=4.*PI*nth*C_e*C_e*v/sqrt(3.)/K_value/C_c;
      double comp2=4.0505/pow(x,(1./6.))*(1.+0.4/pow(x,0.25)+0.5316/sqrt(x))*exp(-1.8899*pow(x,1./3.));
      double j_nu=comp1*comp2;
      double B_nu=2.0*v*v*v*C_h/C_c/C_c/(exp(C_h*v/C_kB/Te)-1.0);

      // integrate intensity along ray
      ds    =  hdid; cpu_transfer_steps++; if(htry != hdid) cpu_transfer_rejected_steps++;

      dtau  =  dtau + ds*C_sgrA_mbh*C_rgeo*j_nu/B_nu*zzz;  
      dI    =  dI + ds*C_sgrA_mbh*C_rgeo*j_nu/freq_local/freq_local/freq_local*exp(-dtau)*zzz;  
    }
    htry = hnext;
  }
}

}

namespace ld14 {
static long double cpu_event[64][3]={};
// Deterministic host-only clone of original CUDA source; qualifiers/index mapping changed, diagnostics added without changing equations.
static int cpu_x,cpu_y; static long long cpu_accepted_steps=0,cpu_shrunk_steps=0,cpu_transfer_steps=0,cpu_transfer_rejected_steps=0; static long double cpu_initial_null_max=0;
/*
  Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
  Odyssey  version 1.0   (released  2015)
  This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
  for General Relativistic Radiative Transfer (GRRT), following the 
  ray-tracing algorithm presented in 
  Fuerst, S. V., & Wu, K. 2007, A&A, 474, 55, 
  and the radiative transfer formulation described in 
  Younsi, Z., Wu, K., & Fuerst, S. V. 2012, A&A, 545, A13

  Odyssey is distributed freely under the GNU general public license. 
  You can redistribute it and/or modify it under the terms of the License
  http://www.gnu.org/licenses/gpl.txt
  The current distribution website is:
  https://github.com/hungyipu/Odyssey/ 
 */


 static void cpu_geodesic(long double* Variables, long double* VariablesIn, long double *y, long double *dydx)
{
  long double r = y[0];
  long double theta = y[1];
  long double pr = y[4];
  long double ptheta = y[5];

  long double r2  = r * r;
  long double twor = 2.0 * r;

  long double sintheta = sin(theta);
  long double costheta = cos(theta);
  long double cos2     = costheta * costheta;
  long double sin2     = sintheta * sintheta;
  long double sigma  = r2 + a2 * cos2;
  long double delta  = r2 - twor + a2;
  long double sd    = sigma * delta;
  long double siginv  = 1.0 / sigma;
  long double bot    = 1.0 / sd;


  //avoid problems near the axis
  if (sintheta < 1e-8)
  {
    sintheta = 1e-8;
    sin2 = 1e-16;
  }

  dydx[0] = -pr * delta * siginv;  
  dydx[1] = -ptheta * siginv;
  dydx[2] = -(twor * A + (sigma - twor) * L / sin2) * bot;
  dydx[3] = -(1.0 + (twor * (r2 + a2) - twor * A * L) * bot);
  dydx[4] = -(((r - 1.0) * (-kappa) + twor * (r2 + a2) - 2.0 * A * L) * bot - 2.0 * pr * pr * (r - 1.0) * siginv);
  dydx[5] = -sintheta * costheta*(L * L / (sin2 * sin2) - a2) * siginv;
}

 static void cpu_rkstep(long double* Variables, long double* VariablesIn,long double *y,
                              long double *dydx, long double h, long double *yout, long double *yerr)
{
  int i;
  long double ak[N];
  long double ytemp1[N], ytemp2[N], ytemp3[N], ytemp4[N], ytemp5[N];
  long double hdydx;
  long double yi, yt;

  for (i = 0; i < N; i++)
  {
    hdydx     = h * dydx[i];
    yi        = y[i];
    ytemp1[i] = yi + 0.2 * hdydx;
    ytemp2[i] = yi + (3.0/40.0) * hdydx;
    ytemp3[i] = yi + 0.3 * hdydx;
    ytemp4[i] = yi -(11.0/54.0) * hdydx;
    ytemp5[i] = yi + (1631.0/55296.0) * hdydx;
    yout[i]   = yi + (37.0/378.0) * hdydx;
    yerr[i]   = ((37.0/378.0)-(2825.0/27648.0)) * hdydx;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp1, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp2[i] += (9.0/40.0) * yt;
    ytemp3[i] -= 0.9 * yt;
    ytemp4[i] += 2.5 * yt;
    ytemp5[i] += (175.0/512.0) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp2, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp3[i] += 1.2 * yt;
    ytemp4[i] -= (70.0/27.0) * yt;
    ytemp5[i] += (575.0/13824.0) * yt;
    yout[i]   += (250.0/621.0) * yt;
    yerr[i]   += ((250.0/621.0)-(18575.0/48384.0)) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp3, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp4[i] += (35.0/27.0) * yt;
    ytemp5[i] += (44275.0/110592.0) * yt;
    yout[i]   += (125.0/594.0) * yt;
    yerr[i]   += ((125.0/594.0)-(13525.0/55296.0)) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp4, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp5[i] += (253.0/4096.0) * yt;
    yerr[i]   -= (277.0/14336.0) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp5, ak);

  for (i = 0; i < N; i++)
  {
    yt       = h * ak[i];
    yout[i] += (512.0/1771.0) * yt;
    yerr[i] += ((512.0/1771.0)-0.25) * yt;
  }
}

 static long double cpu_rk5(long double* Variables, long double* VariablesIn, long double *y, long double *dydx, 
                             long double htry, long double escal, long double *yscal, long double *hdid)
{
  int i;
  long double hnext;
  long double errmax, h = htry, htemp;
  long double yerr[N], ytemp[N];

  while (1)
  {
    // find adaptive step size
    cpu_rkstep(Variables, VariablesIn, y, dydx, h, ytemp, yerr);

    errmax = 0.0;
    for (i = 0; i < N; i++)
    {
      long double temp = fabs(yerr[i]/yscal[i]);
      if (temp > errmax) errmax = temp;
    }

    errmax *= escal;
    if (errmax <= 1.0) break;

    htemp = 0.9 * h / sqrt(sqrt(errmax));

    h *= 0.1;

    if (h >= 0.0)
    {
      if (htemp > h) h = htemp;
    }
    else
    {
      if (htemp < h) h = htemp;
    }
  }

  if (errmax > 1.89e-4)
  {
    hnext = 0.9 * h * pow(errmax, -0.2);
  }
  else
  {
    hnext = 5.0 * h;
  }

  *hdid = h; cpu_accepted_steps++; if(h != htry) cpu_shrunk_steps++;

  memcpy(y, ytemp, N * sizeof(long double));

  return hnext;
}

 static void cpu_initial(long double* Variables, long double* VariablesIn, long double *y0, long double *ydot0)
{
  long double alpha = grid_x;
  long double beta  = grid_y;

  //see Eq[18-19], with phi_obs=0, x= alpha, y=beta, z=0
  long double x     = sqrt(r0*r0+a2)*sin(theta0)-beta*cos(theta0);
  long double y     = alpha;
  long double z     = r0*cos(theta0)+beta*sin(theta0);
  long double w     = x*x+y*y+z*z-a2;

  //see Eq[20-22]
  y0[0] = sqrt((w+sqrt(w*w+(2.*A*z)*(2.*A*z)))/2.);   
  y0[1] = acos(z/y0[0]);                              
  y0[2] = atan2(y,x);                                
  y0[3] = 0;
  long double r = y0[0];
  long double theta = y0[1];
  long double phi=y0[2];

  long double sigma = r*r+(A*cos(theta))*(A*cos(theta));
  long double u     = sqrt(a2+r*r);
  long double v     = -sin(theta0)*cos(phi);
  long double zdot  = -1.;

  //see Eq[24-26]
  long double rdot0 = zdot*(-u*u*cos(theta0)*cos(theta)+r*u*v*sin(theta))/sigma;         
  long double thetadot0 = zdot*(cos(theta0)*r*sin(theta)+u*v*cos(theta))/sigma;          
  long double phidot0 = zdot*sin(theta0)*sin(phi)/(u*sin(theta));                         

  ydot0[0] = rdot0;
  ydot0[1] = thetadot0;
  ydot0[2] = phidot0;

  long double sintheta=sin(theta);
  long double sin2 = sintheta*sintheta;

  long double r2 = r * r;
  long double delta = r2 - 2.0 * r + a2;
  long double s1 = sigma - 2.0 * r;

  y0[4] = rdot0*sigma/delta;
  y0[5] = thetadot0*sigma;

  // Eq.7
  long double energy2 = s1*(rdot0*rdot0/delta+thetadot0*thetadot0)
    + delta*sin2*phidot0*phidot0;

  long double energy = sqrt(energy2);

  // rescaled by energy
  y0[4] = y0[4]/energy;
  y0[5] = y0[5]/energy;

  // Eq.8 
  L = ((sigma*delta*phidot0-2.0*A*r*energy)*sin2/s1)/energy;

  // Eq below Eq.15
  kappa = y0[5]*y0[5]+a2*sin2+L*L/sin2; if(A==0){long double H=-1/(1-2/r)+(1-2/r)*y0[4]*y0[4]+y0[5]*y0[5]/r2+L*L/(r2*sin2);cpu_initial_null_max=fmax(cpu_initial_null_max,fabs(H));}
}

 static float cpu_ISCO(long double* VariablesIn)
{
  long double z1       = 1 + pow(1 - A * A, 1 / 3.0) * pow(1 + A, 1 / 3.0) + pow(1 - A, 1 / 3.0);
  long double z2       = sqrt(3 * A * A + z1 * z1);
  return 3. + z2 - sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

//
// functions for radiative transfer                                                  
//

#define Te_min    0.1
#define Te_max    100.
#define Te_grids  50.

static   long double cpu_K2_tab[] = {
  -10.747001, //Te=0.1
  -9.362569,
  -8.141373,
  -7.061568,
  -6.104060,
  -5.252153,
  -4.491244,
  -3.808555,
  -3.192909,
  -2.634534,
  -2.124893,
  -1.656543,
  -1.223007,
  -0.818668,
  -0.438676,
  -0.078863,
  +0.264332,
  +0.593930,
  +0.912476,
  +1.222098,
  +1.524560,
  +1.821311,
  +2.113537,
  +2.402193,
  +2.688050,
  +2.971721,
  +3.253692,
  +3.534347,
  +3.813984,
  +4.092839,
  +4.371092,
  +4.648884,
  +4.926323,
  +5.203493,
  +5.480457,
  +5.757264,
  +6.033952,
  +6.310550,
  +6.587078,
  +6.863554,
  +7.139990,
  +7.416395,
  +7.692778,
  +7.969143,
  +8.245495,
  +8.521837,
  +8.798171,
  +9.074500,
  +9.350824,
  +9.627144  //Te=87.09
};

 static long double cpu_K2_find(long double Te)
{
  long double d = Te_grids*(log(Te / Te_min)/ log(Te_max / Te_min));
  int    i = floor(d);

  return (1 - (long double)(d-i)) * cpu_K2_tab[i] + (long double)(d-i) * cpu_K2_tab[i+1];
}

 static long double cpu_K2(long double Te)
{
  long double tab_K2;
  //avoid the boundary effect when T~T_max
  if (Te>85.){ 
    tab_K2=2.*Te*Te;
    return tab_K2;
  }

  if (Te<Te_min){ 
    //set a dummy value
    return exp(-11.);
  }

  tab_K2= cpu_K2_find(Te);
  return exp(tab_K2);
}

 static long double cpu_Jansky_Correction(long double* VariablesIn,long double ima_width)
{
  long double distance=C_sgrA_d*C_pc;
  long double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
  long double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //radian^2 (steradians) per pixel
  return pix_str/C_Jansky;
}

 static long double cpu_Luminosity_Correction(long double* VariablesIn,long double ima_width)
{
  long double distance=C_sgrA_d*C_pc;
  long double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
  long double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //radian^2 (steradians) per pixel
  return pix_str*distance*distance*4.*PI*freq_obs;
}

 long double cpu_task1fun_GetZ(long double* Variables, long double* VariablesIn, long double *y)
{
  long double r1 = y[0];
  long double E_local = -(r1 * r1 + A * sqrt(r1)) / (r1 * sqrt(r1 * r1 - 3. * r1 + 2. * A * sqrt(r1))) + 
                  L / sqrt(r1) / sqrt(r1 * r1 - 3. * r1  +2. * A * sqrt(r1));
  long double E_inf = -1.0;      
  return E_local / E_inf; 
}

 void cpu_task1(long double*__restrict__ ResultsPixel,
                      long double*__restrict__ VariablesIn,
                      int GridIdxX, int GridIdxY)
{
  // to check whether the photon is inside image plane
  if(cpu_x >= SIZE || cpu_y >= SIZE) return;  

  long double Variables[VarNUM];

  r0       = 1000.0;                 
  theta0   = (PI/180.0) * INCLINATION;     
  a2       = A * A;                
  Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;      
  Rmstable = cpu_ISCO(VariablesIn);                

  long double htry = 0.5, escal = 1e14, hdid = 0.0, hnext = 0.0;
  long double y[N], dydx[N], yscal[N], ylaststep[N];
  long double Rdisk   = 50.;
  long double ima_width = 55.;
  long double s1  = ima_width;                      
  long double s2  = 2.*ima_width/((int)SIZE+1.);   

  grid_x = -s1 + s2*(cpu_x+1.);
  grid_y = -s1 + s2*(cpu_y+1.);

  cpu_initial(Variables, VariablesIn, y, dydx);

  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+0] = grid_x;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+1] = grid_y;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 0;

  while (1)
  {
    for(int i = 0; i < N; i++) ylaststep[i] = y[i];

    cpu_geodesic(Variables, VariablesIn, y, dydx);

    for (int i = 0; i < N; i++) yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

    //fifth-order Runge-Kutta method
    hnext = cpu_rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

    // hit the disk, compute redshift
    if( y[0] < Rdisk && y[0] > Rmstable && (ylaststep[1] - PI/2.) * (y[1] - PI/2.) < 0. )
    {    
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 1./cpu_task1fun_GetZ(Variables, VariablesIn, y);
      long double fraction=(PI/2.-ylaststep[1])/(y[1]-ylaststep[1]);
      long double plane[N];for(int q=0;q<N;q++)plane[q]=ylaststep[q]+fraction*(y[q]-ylaststep[q]);
      int pix=IMAGE_SIZE*cpu_y+cpu_x;
      cpu_event[pix][0]=ResultsPixel[3*pix+2];
      cpu_event[pix][1]=1./cpu_task1fun_GetZ(Variables,VariablesIn,plane);
      cpu_event[pix][2]=fabs(1./cpu_task1fun_GetZ(Variables,VariablesIn,ylaststep)-cpu_event[pix][0]);

      break;
    }

    // Inside the event horizon radius or escape to infinity
    if ((y[0] > r0) && (dydx[0]>0)) break;

    if (y[0] < Rhor) break;

    htry = hnext;
  }
}

 long double cpu_task2fun_GetZ(long double* Variables, long double* VariablesIn, long double *y)
{
  long double ut,uphi,ur,E_local;
  long double E_inf= -1.0;   
  long double r=y[0];
  long double theta=y[1];
  long double pr=y[4];

  long double r2 = r*r;
  long double twor = 2.0*r;
  long double sintheta, costheta;
  sintheta=sin(theta);
  costheta=cos(theta);
  long double cos2 = costheta*costheta;
  long double sin2 = sintheta*sintheta;

  long double sigma = r2+a2*cos2;
  long double delta = r2-twor+a2;
  long double ssig=(r2+a2)*(r2+a2)-a2*delta*sin2; 

  //======define (covariant) metric component    
  long double gtt=-(1.-2.*r/sigma);
  long double gtph=-2.*A*r*sin2/sigma;
  long double grr=sigma/delta;
  long double gphph=ssig*sin2/sigma;    


  //==========Keplerian flow: outside cpu_ISCO
  long double ut_k  =(r*r+A*sqrt(r))/(r*sqrt(r*r-3.*r+2.*A*sqrt(r)));
  long double ur_k  =0.;
  long double uphi_k  =1./(sqrt(r)*sqrt(r*r-3.*r+2.*A*sqrt(r)));

  //==========Keplerian flow: inside cpu_ISCO    
  if( r<Rmstable)
  {

    long double delta = r*r-2.*r+a2;
    long double lambda=(Rmstable*Rmstable-2.*A*sqrt(Rmstable)+a2)/(sqrt(Rmstable*Rmstable*Rmstable)-2.*sqrt(Rmstable)+A);
    long double gamma=sqrt(1-2./3./Rmstable);
    long double h=(2.*r-A*lambda)/delta;

    ut_k=gamma*(1.+2/r*(1.+h));
    ur_k=-sqrt(2./3./Rmstable)*sqrt(pow((Rmstable/r-1.),3.));
    uphi_k=gamma/r/r*(lambda+A*h);
  }  

  //normalize the four-velocity
  ut    = ut_k;
  uphi  = uphi_k;
  ur    = ur_k;  
  long double omega = uphi/ut;
  long double k0    = -(gtt + omega*omega*gphph + 2.*omega*gtph);
  ut = sqrt(((1. + grr*ur*ur) / k0));
  uphi = omega*ut;  

  // compute redshift
  E_local=-ut+L*uphi+pr*ur;
  return E_local/E_inf; 
}

 void cpu_task2(long double*__restrict__ ResultsPixel,
                      long double*__restrict__ VariablesIn,
                      int GridIdxX, int GridIdxY)
{
  if(cpu_x >= SIZE || cpu_y >= SIZE) return;  

  long double Variables[VarNUM];
  r0       = 1000.0;                 
  theta0   = (PI/180.0) * INCLINATION;     
  a2       = A * A;                
  Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;   
  Rmstable = cpu_ISCO(VariablesIn);  

  long double htry = 0.5, escal = 1e14, hdid = 0.0, hnext = 0.0;
  long double y[N], dydx[N], yscal[N];

  long double Rdisk   = 500.;
  long double ima_width = 10.;
  long double s1  = ima_width;                      
  long double s2  = 2.*ima_width/((int)SIZE+1.);   
  long double Jy_corr=cpu_Jansky_Correction(VariablesIn,ima_width);
  long double L_corr=cpu_Luminosity_Correction(VariablesIn,ima_width);

  grid_x = -s1 + s2*(cpu_x+1.);
  grid_y = -s1 + s2*(cpu_y+1.);

  cpu_initial(Variables, VariablesIn, y, dydx);

  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+0] = grid_x;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+1] = grid_y;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 0;

  long double ds=0.;  
  long double dtau=0.;
  long double dI=0.;

  while (1)
  {
    cpu_geodesic(Variables, VariablesIn, y, dydx);

    for (int i = 0; i < N; i++)
      yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

    hnext = cpu_rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

    if ((y[0] > r0) && (dydx[0]>0)){
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = dI*freq_obs*freq_obs*freq_obs*L_corr; 
      break;
    }

    if (y[0] < Rhor){
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = dI*freq_obs*freq_obs*freq_obs*L_corr; 
      break;
    }

    long double r=y[0];
    long double theta=y[1];

    if(y[0]<Rdisk){

      long double zzz        = cpu_task2fun_GetZ(Variables, VariablesIn, y); //zzz=E_em/E_obs
      long double freq_local = freq_obs*zzz;  

      // thermal synchrotron
      long double nth0=3e7;
      long double zc=r*cos(theta);
      long double rc=r*sin(theta);

      long double nth=nth0*exp(-zc*zc/2./rc/rc)*pow(r,-1.1);
      long double Te=1.7e11*pow(r,-0.84); 
      long double b=sqrt(8.*PI*0.1*nth*C_mp*C_c*C_c/6./r);

      long double vb=C_e*b/2./PI/C_me/C_c;
      long double theta_E= C_kB*Te/C_me/C_c/C_c;
      long double v=freq_local;
      long double x=2.*v/3./vb/theta_E/theta_E;

      long double K_value=cpu_K2(theta_E);

      long double comp1=4.*PI*nth*C_e*C_e*v/sqrt(3.)/K_value/C_c;
      long double comp2=4.0505/pow(x,(1./6.))*(1.+0.4/pow(x,0.25)+0.5316/sqrt(x))*exp(-1.8899*pow(x,1./3.));
      long double j_nu=comp1*comp2;
      long double B_nu=2.0*v*v*v*C_h/C_c/C_c/(exp(C_h*v/C_kB/Te)-1.0);

      // integrate intensity along ray
      ds    =  hdid; cpu_transfer_steps++; if(htry != hdid) cpu_transfer_rejected_steps++;

      dtau  =  dtau + ds*C_sgrA_mbh*C_rgeo*j_nu/B_nu*zzz;  
      dI    =  dI + ds*C_sgrA_mbh*C_rgeo*j_nu/freq_local/freq_local/freq_local*exp(-dtau)*zzz;  
    }
    htry = hnext;
  }
}

}

namespace ld15 {
static long double cpu_event[64][3]={};
// Deterministic host-only clone of original CUDA source; qualifiers/index mapping changed, diagnostics added without changing equations.
static int cpu_x,cpu_y; static long long cpu_accepted_steps=0,cpu_shrunk_steps=0,cpu_transfer_steps=0,cpu_transfer_rejected_steps=0; static long double cpu_initial_null_max=0;
/*
  Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
  Odyssey  version 1.0   (released  2015)
  This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
  for General Relativistic Radiative Transfer (GRRT), following the 
  ray-tracing algorithm presented in 
  Fuerst, S. V., & Wu, K. 2007, A&A, 474, 55, 
  and the radiative transfer formulation described in 
  Younsi, Z., Wu, K., & Fuerst, S. V. 2012, A&A, 545, A13

  Odyssey is distributed freely under the GNU general public license. 
  You can redistribute it and/or modify it under the terms of the License
  http://www.gnu.org/licenses/gpl.txt
  The current distribution website is:
  https://github.com/hungyipu/Odyssey/ 
 */


 static void cpu_geodesic(long double* Variables, long double* VariablesIn, long double *y, long double *dydx)
{
  long double r = y[0];
  long double theta = y[1];
  long double pr = y[4];
  long double ptheta = y[5];

  long double r2  = r * r;
  long double twor = 2.0 * r;

  long double sintheta = sin(theta);
  long double costheta = cos(theta);
  long double cos2     = costheta * costheta;
  long double sin2     = sintheta * sintheta;
  long double sigma  = r2 + a2 * cos2;
  long double delta  = r2 - twor + a2;
  long double sd    = sigma * delta;
  long double siginv  = 1.0 / sigma;
  long double bot    = 1.0 / sd;


  //avoid problems near the axis
  if (sintheta < 1e-8)
  {
    sintheta = 1e-8;
    sin2 = 1e-16;
  }

  dydx[0] = -pr * delta * siginv;  
  dydx[1] = -ptheta * siginv;
  dydx[2] = -(twor * A + (sigma - twor) * L / sin2) * bot;
  dydx[3] = -(1.0 + (twor * (r2 + a2) - twor * A * L) * bot);
  dydx[4] = -(((r - 1.0) * (-kappa) + twor * (r2 + a2) - 2.0 * A * L) * bot - 2.0 * pr * pr * (r - 1.0) * siginv);
  dydx[5] = -sintheta * costheta*(L * L / (sin2 * sin2) - a2) * siginv;
}

 static void cpu_rkstep(long double* Variables, long double* VariablesIn,long double *y,
                              long double *dydx, long double h, long double *yout, long double *yerr)
{
  int i;
  long double ak[N];
  long double ytemp1[N], ytemp2[N], ytemp3[N], ytemp4[N], ytemp5[N];
  long double hdydx;
  long double yi, yt;

  for (i = 0; i < N; i++)
  {
    hdydx     = h * dydx[i];
    yi        = y[i];
    ytemp1[i] = yi + 0.2 * hdydx;
    ytemp2[i] = yi + (3.0/40.0) * hdydx;
    ytemp3[i] = yi + 0.3 * hdydx;
    ytemp4[i] = yi -(11.0/54.0) * hdydx;
    ytemp5[i] = yi + (1631.0/55296.0) * hdydx;
    yout[i]   = yi + (37.0/378.0) * hdydx;
    yerr[i]   = ((37.0/378.0)-(2825.0/27648.0)) * hdydx;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp1, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp2[i] += (9.0/40.0) * yt;
    ytemp3[i] -= 0.9 * yt;
    ytemp4[i] += 2.5 * yt;
    ytemp5[i] += (175.0/512.0) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp2, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp3[i] += 1.2 * yt;
    ytemp4[i] -= (70.0/27.0) * yt;
    ytemp5[i] += (575.0/13824.0) * yt;
    yout[i]   += (250.0/621.0) * yt;
    yerr[i]   += ((250.0/621.0)-(18575.0/48384.0)) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp3, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp4[i] += (35.0/27.0) * yt;
    ytemp5[i] += (44275.0/110592.0) * yt;
    yout[i]   += (125.0/594.0) * yt;
    yerr[i]   += ((125.0/594.0)-(13525.0/55296.0)) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp4, ak);

  for (i = 0; i < N; i++)
  {
    yt         = h * ak[i];
    ytemp5[i] += (253.0/4096.0) * yt;
    yerr[i]   -= (277.0/14336.0) * yt;
  }

  cpu_geodesic(Variables, VariablesIn, ytemp5, ak);

  for (i = 0; i < N; i++)
  {
    yt       = h * ak[i];
    yout[i] += (512.0/1771.0) * yt;
    yerr[i] += ((512.0/1771.0)-0.25) * yt;
  }
}

 static long double cpu_rk5(long double* Variables, long double* VariablesIn, long double *y, long double *dydx, 
                             long double htry, long double escal, long double *yscal, long double *hdid)
{
  int i;
  long double hnext;
  long double errmax, h = htry, htemp;
  long double yerr[N], ytemp[N];

  while (1)
  {
    // find adaptive step size
    cpu_rkstep(Variables, VariablesIn, y, dydx, h, ytemp, yerr);

    errmax = 0.0;
    for (i = 0; i < N; i++)
    {
      long double temp = fabs(yerr[i]/yscal[i]);
      if (temp > errmax) errmax = temp;
    }

    errmax *= escal;
    if (errmax <= 1.0) break;

    htemp = 0.9 * h / sqrt(sqrt(errmax));

    h *= 0.1;

    if (h >= 0.0)
    {
      if (htemp > h) h = htemp;
    }
    else
    {
      if (htemp < h) h = htemp;
    }
  }

  if (errmax > 1.89e-4)
  {
    hnext = 0.9 * h * pow(errmax, -0.2);
  }
  else
  {
    hnext = 5.0 * h;
  }

  *hdid = h; cpu_accepted_steps++; if(h != htry) cpu_shrunk_steps++;

  memcpy(y, ytemp, N * sizeof(long double));

  return hnext;
}

 static void cpu_initial(long double* Variables, long double* VariablesIn, long double *y0, long double *ydot0)
{
  long double alpha = grid_x;
  long double beta  = grid_y;

  //see Eq[18-19], with phi_obs=0, x= alpha, y=beta, z=0
  long double x     = sqrt(r0*r0+a2)*sin(theta0)-beta*cos(theta0);
  long double y     = alpha;
  long double z     = r0*cos(theta0)+beta*sin(theta0);
  long double w     = x*x+y*y+z*z-a2;

  //see Eq[20-22]
  y0[0] = sqrt((w+sqrt(w*w+(2.*A*z)*(2.*A*z)))/2.);   
  y0[1] = acos(z/y0[0]);                              
  y0[2] = atan2(y,x);                                
  y0[3] = 0;
  long double r = y0[0];
  long double theta = y0[1];
  long double phi=y0[2];

  long double sigma = r*r+(A*cos(theta))*(A*cos(theta));
  long double u     = sqrt(a2+r*r);
  long double v     = -sin(theta0)*cos(phi);
  long double zdot  = -1.;

  //see Eq[24-26]
  long double rdot0 = zdot*(-u*u*cos(theta0)*cos(theta)+r*u*v*sin(theta))/sigma;         
  long double thetadot0 = zdot*(cos(theta0)*r*sin(theta)+u*v*cos(theta))/sigma;          
  long double phidot0 = zdot*sin(theta0)*sin(phi)/(u*sin(theta));                         

  ydot0[0] = rdot0;
  ydot0[1] = thetadot0;
  ydot0[2] = phidot0;

  long double sintheta=sin(theta);
  long double sin2 = sintheta*sintheta;

  long double r2 = r * r;
  long double delta = r2 - 2.0 * r + a2;
  long double s1 = sigma - 2.0 * r;

  y0[4] = rdot0*sigma/delta;
  y0[5] = thetadot0*sigma;

  // Eq.7
  long double energy2 = s1*(rdot0*rdot0/delta+thetadot0*thetadot0)
    + delta*sin2*phidot0*phidot0;

  long double energy = sqrt(energy2);

  // rescaled by energy
  y0[4] = y0[4]/energy;
  y0[5] = y0[5]/energy;

  // Eq.8 
  L = ((sigma*delta*phidot0-2.0*A*r*energy)*sin2/s1)/energy;

  // Eq below Eq.15
  kappa = y0[5]*y0[5]+a2*sin2+L*L/sin2; if(A==0){long double H=-1/(1-2/r)+(1-2/r)*y0[4]*y0[4]+y0[5]*y0[5]/r2+L*L/(r2*sin2);cpu_initial_null_max=fmax(cpu_initial_null_max,fabs(H));}
}

 static float cpu_ISCO(long double* VariablesIn)
{
  long double z1       = 1 + pow(1 - A * A, 1 / 3.0) * pow(1 + A, 1 / 3.0) + pow(1 - A, 1 / 3.0);
  long double z2       = sqrt(3 * A * A + z1 * z1);
  return 3. + z2 - sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

//
// functions for radiative transfer                                                  
//

#define Te_min    0.1
#define Te_max    100.
#define Te_grids  50.

static   long double cpu_K2_tab[] = {
  -10.747001, //Te=0.1
  -9.362569,
  -8.141373,
  -7.061568,
  -6.104060,
  -5.252153,
  -4.491244,
  -3.808555,
  -3.192909,
  -2.634534,
  -2.124893,
  -1.656543,
  -1.223007,
  -0.818668,
  -0.438676,
  -0.078863,
  +0.264332,
  +0.593930,
  +0.912476,
  +1.222098,
  +1.524560,
  +1.821311,
  +2.113537,
  +2.402193,
  +2.688050,
  +2.971721,
  +3.253692,
  +3.534347,
  +3.813984,
  +4.092839,
  +4.371092,
  +4.648884,
  +4.926323,
  +5.203493,
  +5.480457,
  +5.757264,
  +6.033952,
  +6.310550,
  +6.587078,
  +6.863554,
  +7.139990,
  +7.416395,
  +7.692778,
  +7.969143,
  +8.245495,
  +8.521837,
  +8.798171,
  +9.074500,
  +9.350824,
  +9.627144  //Te=87.09
};

 static long double cpu_K2_find(long double Te)
{
  long double d = Te_grids*(log(Te / Te_min)/ log(Te_max / Te_min));
  int    i = floor(d);

  return (1 - (long double)(d-i)) * cpu_K2_tab[i] + (long double)(d-i) * cpu_K2_tab[i+1];
}

 static long double cpu_K2(long double Te)
{
  long double tab_K2;
  //avoid the boundary effect when T~T_max
  if (Te>85.){ 
    tab_K2=2.*Te*Te;
    return tab_K2;
  }

  if (Te<Te_min){ 
    //set a dummy value
    return exp(-11.);
  }

  tab_K2= cpu_K2_find(Te);
  return exp(tab_K2);
}

 static long double cpu_Jansky_Correction(long double* VariablesIn,long double ima_width)
{
  long double distance=C_sgrA_d*C_pc;
  long double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
  long double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //radian^2 (steradians) per pixel
  return pix_str/C_Jansky;
}

 static long double cpu_Luminosity_Correction(long double* VariablesIn,long double ima_width)
{
  long double distance=C_sgrA_d*C_pc;
  long double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
  long double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //radian^2 (steradians) per pixel
  return pix_str*distance*distance*4.*PI*freq_obs;
}

 long double cpu_task1fun_GetZ(long double* Variables, long double* VariablesIn, long double *y)
{
  long double r1 = y[0];
  long double E_local = -(r1 * r1 + A * sqrt(r1)) / (r1 * sqrt(r1 * r1 - 3. * r1 + 2. * A * sqrt(r1))) + 
                  L / sqrt(r1) / sqrt(r1 * r1 - 3. * r1  +2. * A * sqrt(r1));
  long double E_inf = -1.0;      
  return E_local / E_inf; 
}

 void cpu_task1(long double*__restrict__ ResultsPixel,
                      long double*__restrict__ VariablesIn,
                      int GridIdxX, int GridIdxY)
{
  // to check whether the photon is inside image plane
  if(cpu_x >= SIZE || cpu_y >= SIZE) return;  

  long double Variables[VarNUM];

  r0       = 1000.0;                 
  theta0   = (PI/180.0) * INCLINATION;     
  a2       = A * A;                
  Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;      
  Rmstable = cpu_ISCO(VariablesIn);                

  long double htry = 0.5, escal = 1e15, hdid = 0.0, hnext = 0.0;
  long double y[N], dydx[N], yscal[N], ylaststep[N];
  long double Rdisk   = 50.;
  long double ima_width = 55.;
  long double s1  = ima_width;                      
  long double s2  = 2.*ima_width/((int)SIZE+1.);   

  grid_x = -s1 + s2*(cpu_x+1.);
  grid_y = -s1 + s2*(cpu_y+1.);

  cpu_initial(Variables, VariablesIn, y, dydx);

  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+0] = grid_x;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+1] = grid_y;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 0;

  while (1)
  {
    for(int i = 0; i < N; i++) ylaststep[i] = y[i];

    cpu_geodesic(Variables, VariablesIn, y, dydx);

    for (int i = 0; i < N; i++) yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

    //fifth-order Runge-Kutta method
    hnext = cpu_rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

    // hit the disk, compute redshift
    if( y[0] < Rdisk && y[0] > Rmstable && (ylaststep[1] - PI/2.) * (y[1] - PI/2.) < 0. )
    {    
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 1./cpu_task1fun_GetZ(Variables, VariablesIn, y);
      long double fraction=(PI/2.-ylaststep[1])/(y[1]-ylaststep[1]);
      long double plane[N];for(int q=0;q<N;q++)plane[q]=ylaststep[q]+fraction*(y[q]-ylaststep[q]);
      int pix=IMAGE_SIZE*cpu_y+cpu_x;
      cpu_event[pix][0]=ResultsPixel[3*pix+2];
      cpu_event[pix][1]=1./cpu_task1fun_GetZ(Variables,VariablesIn,plane);
      cpu_event[pix][2]=fabs(1./cpu_task1fun_GetZ(Variables,VariablesIn,ylaststep)-cpu_event[pix][0]);

      break;
    }

    // Inside the event horizon radius or escape to infinity
    if ((y[0] > r0) && (dydx[0]>0)) break;

    if (y[0] < Rhor) break;

    htry = hnext;
  }
}

 long double cpu_task2fun_GetZ(long double* Variables, long double* VariablesIn, long double *y)
{
  long double ut,uphi,ur,E_local;
  long double E_inf= -1.0;   
  long double r=y[0];
  long double theta=y[1];
  long double pr=y[4];

  long double r2 = r*r;
  long double twor = 2.0*r;
  long double sintheta, costheta;
  sintheta=sin(theta);
  costheta=cos(theta);
  long double cos2 = costheta*costheta;
  long double sin2 = sintheta*sintheta;

  long double sigma = r2+a2*cos2;
  long double delta = r2-twor+a2;
  long double ssig=(r2+a2)*(r2+a2)-a2*delta*sin2; 

  //======define (covariant) metric component    
  long double gtt=-(1.-2.*r/sigma);
  long double gtph=-2.*A*r*sin2/sigma;
  long double grr=sigma/delta;
  long double gphph=ssig*sin2/sigma;    


  //==========Keplerian flow: outside cpu_ISCO
  long double ut_k  =(r*r+A*sqrt(r))/(r*sqrt(r*r-3.*r+2.*A*sqrt(r)));
  long double ur_k  =0.;
  long double uphi_k  =1./(sqrt(r)*sqrt(r*r-3.*r+2.*A*sqrt(r)));

  //==========Keplerian flow: inside cpu_ISCO    
  if( r<Rmstable)
  {

    long double delta = r*r-2.*r+a2;
    long double lambda=(Rmstable*Rmstable-2.*A*sqrt(Rmstable)+a2)/(sqrt(Rmstable*Rmstable*Rmstable)-2.*sqrt(Rmstable)+A);
    long double gamma=sqrt(1-2./3./Rmstable);
    long double h=(2.*r-A*lambda)/delta;

    ut_k=gamma*(1.+2/r*(1.+h));
    ur_k=-sqrt(2./3./Rmstable)*sqrt(pow((Rmstable/r-1.),3.));
    uphi_k=gamma/r/r*(lambda+A*h);
  }  

  //normalize the four-velocity
  ut    = ut_k;
  uphi  = uphi_k;
  ur    = ur_k;  
  long double omega = uphi/ut;
  long double k0    = -(gtt + omega*omega*gphph + 2.*omega*gtph);
  ut = sqrt(((1. + grr*ur*ur) / k0));
  uphi = omega*ut;  

  // compute redshift
  E_local=-ut+L*uphi+pr*ur;
  return E_local/E_inf; 
}

 void cpu_task2(long double*__restrict__ ResultsPixel,
                      long double*__restrict__ VariablesIn,
                      int GridIdxX, int GridIdxY)
{
  if(cpu_x >= SIZE || cpu_y >= SIZE) return;  

  long double Variables[VarNUM];
  r0       = 1000.0;                 
  theta0   = (PI/180.0) * INCLINATION;     
  a2       = A * A;                
  Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;   
  Rmstable = cpu_ISCO(VariablesIn);  

  long double htry = 0.5, escal = 1e15, hdid = 0.0, hnext = 0.0;
  long double y[N], dydx[N], yscal[N];

  long double Rdisk   = 500.;
  long double ima_width = 10.;
  long double s1  = ima_width;                      
  long double s2  = 2.*ima_width/((int)SIZE+1.);   
  long double Jy_corr=cpu_Jansky_Correction(VariablesIn,ima_width);
  long double L_corr=cpu_Luminosity_Correction(VariablesIn,ima_width);

  grid_x = -s1 + s2*(cpu_x+1.);
  grid_y = -s1 + s2*(cpu_y+1.);

  cpu_initial(Variables, VariablesIn, y, dydx);

  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+0] = grid_x;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+1] = grid_y;
  ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = 0;

  long double ds=0.;  
  long double dtau=0.;
  long double dI=0.;

  while (1)
  {
    cpu_geodesic(Variables, VariablesIn, y, dydx);

    for (int i = 0; i < N; i++)
      yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

    hnext = cpu_rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

    if ((y[0] > r0) && (dydx[0]>0)){
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = dI*freq_obs*freq_obs*freq_obs*L_corr; 
      break;
    }

    if (y[0] < Rhor){
      ResultsPixel[3*(IMAGE_SIZE*cpu_y+cpu_x)+2] = dI*freq_obs*freq_obs*freq_obs*L_corr; 
      break;
    }

    long double r=y[0];
    long double theta=y[1];

    if(y[0]<Rdisk){

      long double zzz        = cpu_task2fun_GetZ(Variables, VariablesIn, y); //zzz=E_em/E_obs
      long double freq_local = freq_obs*zzz;  

      // thermal synchrotron
      long double nth0=3e7;
      long double zc=r*cos(theta);
      long double rc=r*sin(theta);

      long double nth=nth0*exp(-zc*zc/2./rc/rc)*pow(r,-1.1);
      long double Te=1.7e11*pow(r,-0.84); 
      long double b=sqrt(8.*PI*0.1*nth*C_mp*C_c*C_c/6./r);

      long double vb=C_e*b/2./PI/C_me/C_c;
      long double theta_E= C_kB*Te/C_me/C_c/C_c;
      long double v=freq_local;
      long double x=2.*v/3./vb/theta_E/theta_E;

      long double K_value=cpu_K2(theta_E);

      long double comp1=4.*PI*nth*C_e*C_e*v/sqrt(3.)/K_value/C_c;
      long double comp2=4.0505/pow(x,(1./6.))*(1.+0.4/pow(x,0.25)+0.5316/sqrt(x))*exp(-1.8899*pow(x,1./3.));
      long double j_nu=comp1*comp2;
      long double B_nu=2.0*v*v*v*C_h/C_c/C_c/(exp(C_h*v/C_kB/Te)-1.0);

      // integrate intensity along ray
      ds    =  hdid; cpu_transfer_steps++; if(htry != hdid) cpu_transfer_rejected_steps++;

      dtau  =  dtau + ds*C_sgrA_mbh*C_rgeo*j_nu/B_nu*zzz;  
      dI    =  dI + ds*C_sgrA_mbh*C_rgeo*j_nu/freq_local/freq_local/freq_local*exp(-dtau)*zzz;  
    }
    htry = hnext;
  }
}

}

template<typename T,typename F1,typename F2>void run(const char*name,F1 task1,F2 task2,int&x,int&y,T event[64][3],long long&accepted,long long&shrunk){T input[4]={(T)0,(T)(acos(.25)/PI*180),(T)8,(T)340e9},out[192];std::vector<double>point,interp,span,lum;for(y=0;y<8;y++)for(x=0;x<8;x++)task1(out,input,0,0);for(int i=0;i<64;i++){point.push_back((double)out[3*i+2]);interp.push_back((double)event[i][1]);span.push_back((double)event[i][2]);}input[1]=45;for(y=0;y<8;y++)for(x=0;x<8;x++)task2(out,input,0,0);for(int i=0;i<64;i++)lum.push_back((double)out[3*i+2]);printf("{\"case\":\"%s\",\"accepted_steps\":%lld,\"shrunk_steps\":%lld",name,accepted,shrunk);const char*keys[]={"task1_point","task1_linear_plane","task1_crossing_span","task2_luminosity"};std::vector<double>*vectors[]={&point,&interp,&span,&lum};for(int k=0;k<4;k++){printf(",\"%s\":[",keys[k]);for(int i=0;i<64;i++)printf("%s%.17g",i?",":"",(*vectors[k])[i]);printf("]");}printf("}\n");}
int main(){
run("d12",d12::cpu_task1,d12::cpu_task2,d12::cpu_x,d12::cpu_y,d12::cpu_event,d12::cpu_accepted_steps,d12::cpu_shrunk_steps);
run("d14",d14::cpu_task1,d14::cpu_task2,d14::cpu_x,d14::cpu_y,d14::cpu_event,d14::cpu_accepted_steps,d14::cpu_shrunk_steps);
run("d15",d15::cpu_task1,d15::cpu_task2,d15::cpu_x,d15::cpu_y,d15::cpu_event,d15::cpu_accepted_steps,d15::cpu_shrunk_steps);
run("ld14",ld14::cpu_task1,ld14::cpu_task2,ld14::cpu_x,ld14::cpu_y,ld14::cpu_event,ld14::cpu_accepted_steps,ld14::cpu_shrunk_steps);
run("ld15",ld15::cpu_task1,ld15::cpu_task2,ld15::cpu_x,ld15::cpu_y,ld15::cpu_event,ld15::cpu_accepted_steps,ld15::cpu_shrunk_steps);
}
