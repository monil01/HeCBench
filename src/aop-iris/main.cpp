#include <iris/iris.hpp>
#include "payoff.hpp"
#include "gpu_lock.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <vector>
void submit(const std::string& name,size_t blocks,size_t threads,int argc,void**params,int*info){
 iris::Task task;size_t global=blocks*threads,local=threads;
 task.kernel(name.c_str(),1,nullptr,&global,&local,argc,params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("AOP launch failed "+name);
}
template<class Payoff> void do_run(double* samples,int steps,int n,const Payoff& p,double dt,double S0,double r,double sigma,iris::DMem& msamples,iris::DMem& paths,iris::DMem& cashflows,iris::DMem& svds,iris::DMem& skip,iris::DMem& temp,double*price){
 GpuLock lock;
 msamples.update(samples);
 std::string kind=std::is_same<Payoff,PayoffPut>::value?"put":"call";
 {void*q[]={&steps,&n,const_cast<Payoff*>(&p),&dt,&S0,&r,&sigma,&msamples,&paths};int info[]={sizeof(steps),sizeof(n),sizeof(p),sizeof(dt),sizeof(S0),sizeof(r),sizeof(sigma),iris_r,iris_rw};submit("generate_"+kind,(n+255)/256,256,9,q,info);}
 // Original all-out-of-money reset is a DMem host update.
 static std::vector<int> flags;flags.assign(steps,0);skip.update(flags.data());
 int minimum=4;
 {void*q[]={&n,&minimum,const_cast<Payoff*>(&p),&paths,&skip,&svds};int info[]={sizeof(n),sizeof(minimum),sizeof(p),iris_r,iris_rw,iris_rw};submit("svd_"+kind,steps-1,256,6,q,info);}
 double discount=std::exp(-r*dt);int cash_offset=(steps-1)*n,grid=(n+127)/128;
 const int wave=256*112;double waves=grid*128/double(wave);int updategrid=grid;if(waves<10&&waves-int(waves)<0.6)updategrid=std::max(1,int(waves))*wave/128;
 for(int t=steps-2;t>=0;t--){
  {void*q[]={&n,const_cast<Payoff*>(&p),&svds,&paths,&skip,&temp,&t,&cash_offset};int info[]={sizeof(n),sizeof(p),iris_r,iris_r,iris_r,iris_rw,sizeof(t),sizeof(cash_offset)};submit("partial_"+kind,128,128,8,q,info);}
  {void*q[]={&skip,&temp,&t};int info[]={iris_r,iris_rw,sizeof(t)};submit("final_beta",1,128,3,q,info);}
  {void*q[]={&n,const_cast<Payoff*>(&p),&discount,&temp,&paths,&skip,&t,&cash_offset};int info[]={sizeof(n),sizeof(p),sizeof(discount),iris_r,iris_rw,iris_r,sizeof(t),sizeof(cash_offset)};submit("update_"+kind,updategrid,128,8,q,info);}
 }
 {void*q[]={&n,&paths,&temp,&cash_offset};int info[]={sizeof(n),iris_r,iris_rw,sizeof(cash_offset)};submit("partial_sums",grid,128,4,q,info);}
 {void*q[]={&n,&grid,&discount,&temp};int info[]={sizeof(n),sizeof(grid),sizeof(discount),iris_rw};submit("final_sum",1,128,4,q,info);}
 iris::Task output;output.flush_out(temp);output.submit(iris_gpu,nullptr,1);
 std::vector<double> results(4*2048);
 temp.fetch(results.data());*price=results[0];
 if(!std::isfinite(*price))throw std::runtime_error("nonfinite option price");
}
template< typename Payoff >
static double binomial_tree(int num_timesteps, const Payoff &payoff, double dt, double S0, double r, double sigma)
{
  double *tree = new double[num_timesteps+1];

  double u = std::exp( sigma * std::sqrt(dt));
  double d = std::exp(-sigma * std::sqrt(dt));
  double a = std::exp( r     * dt);

  double p = (a - d) / (u - d);

  double k = std::pow(d, num_timesteps);
  for( int t = 0 ; t <= num_timesteps ; ++t )
  {
    tree[t] = payoff(S0*k);
    k *= u*u;
  }

  for( int t = num_timesteps-1 ; t >= 0 ; --t )
  {
    k = std::pow(d, t);
    for( int i = 0 ; i <= t ; ++i )
    {
      double expected = std::exp(-r*dt) * (p*tree[i+1] + (1.0 - p)*tree[i]);
      double earlyex = payoff(S0*k);
      tree[i] = std::max(earlyex, expected);
      k *= u*u;
    }
  }

  double f = tree[0];
  delete[] tree;
  return f;
}

// Calculate the standard normal cumulative distribution function
inline double my_normcdf (double x) {
  return (1.0 + erf(x / sqrt(2.0))) / 2.0;
}

static double black_scholes_merton_put(double T, double K, double S0, double r, double sigma)
{
  double d1 = (std::log(S0 / K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
  double d2 = d1 - sigma*std::sqrt(T);

  return K*std::exp(-r*T)*my_normcdf(-d2) - S0*my_normcdf(-d1);
}

static double black_scholes_merton_call(double T, double K, double S0, double r, double sigma)
{
  double d1 = (std::log(S0 / K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
  double d2 = d1 - sigma*std::sqrt(T);

  return S0*my_normcdf(d1) - K*std::exp(-r*T)*my_normcdf(d2);
}

int main(int argc, char **argv)
{
  iris::Platform platform;{GpuLock lock;platform.init(&argc,&argv,1);}
  int devices=0;iris_device_count(&devices);for(int d=0;d<devices;d++){int b=0;size_t z=0;iris_device_info(d,iris_backend,&b,&z);printf("IRIS backend=%d HIP=%d\n",b,iris_hip);}
  const int MAX_GRID_SIZE = 2048;

  // Simulation parameters.
  int num_timesteps = 100;
  int num_paths     = 32;
  int num_runs      = 1;

  // Option parameters.
  double T     = 1.00;
  double K     = 4.00;
  double S0    = 3.60;
  double r     = 0.06;
  double sigma = 0.20;

  // Bool do we price a put or a call.
  bool price_put = true;

  // Read command-line options.
  for( int i = 1 ; i < argc ; ++i )
  {
    if( !strcmp(argv[i], "-timesteps") )
      num_timesteps = strtol(argv[++i], NULL, 10);
    else if( !strcmp(argv[i], "-paths") )
      num_paths = strtol(argv[++i], NULL, 10);
    else if( !strcmp(argv[i], "-runs") )
      num_runs = strtol(argv[++i], NULL, 10);
    else if( !strcmp(argv[i], "-T") )
      T = strtod(argv[++i], NULL);
    else if( !strcmp(argv[i], "-S0") )
      S0 = strtod(argv[++i], NULL);
    else if( !strcmp(argv[i], "-K") )
      K = strtod(argv[++i], NULL);
    else if( !strcmp(argv[i], "-r") )
      r = strtod(argv[++i], NULL);
    else if( !strcmp(argv[i], "-sigma") )
      sigma = strtod(argv[++i], NULL);
    else if( !strcmp(argv[i], "-call") )
      price_put = false;
    else
    {
      fprintf(stderr, "Unknown option %s. Aborting!!!\n", argv[i]);
      exit(1);
    }
  }

  // Print the arguments.
  printf("==============\n");
  printf("Num Timesteps         : %d\n",  num_timesteps);
  printf("Num Paths             : %dK\n", num_paths);
  printf("Num Runs              : %d\n",  num_runs);
  printf("T                     : %lf\n", T);
  printf("S0                    : %lf\n", S0);
  printf("K                     : %lf\n", K);
  printf("r                     : %lf\n", r);
  printf("sigma                 : %lf\n", sigma);
  printf("Option Type           : American %s\n",  price_put ? "Put" : "Call");

  // We want x1024 paths.
  num_paths *= 1024;

  // A timestep.
  double dt = T / num_timesteps;

  // Generate random samples on a host
  std::default_random_engine rng;
  std::normal_distribution<double> norm_dist(0.0, 1.0);

  double *h_samples = (double*) malloc (num_timesteps*num_paths*sizeof(double));

  std::vector<double> paths(size_t(num_timesteps)*num_paths),svds(16*num_timesteps),temp(4*MAX_GRID_SIZE);
  std::vector<int> skip(num_timesteps);
  iris::DMem d_samples(h_samples,size_t(num_timesteps)*num_paths*sizeof(double)),d_paths(paths.data(),paths.size()*sizeof(double)),d_svds(svds.data(),svds.size()*sizeof(double)),d_all_out_of_the_money(skip.data(),skip.size()*sizeof(int)),d_temp_storage(temp.data(),temp.size()*sizeof(double));
  iris::DMem& d_cashflows=d_paths;
  // The price on the host.
  double h_price;

  // time the do_run function
  float total_elapsed_time = 0;

  for( int run = 0; run < num_runs; ++run )
  {
    for (int i = 0; i < num_timesteps*num_paths; ++i)
      h_samples[i] = norm_dist(rng);

    auto start = std::chrono::high_resolution_clock::now();
    if( price_put )
      do_run(h_samples,
             num_timesteps,
             num_paths,
             PayoffPut(K),
             dt,
             S0,
             r,
             sigma,
             d_samples,
             d_paths,
             d_cashflows,
             d_svds,
             d_all_out_of_the_money,
             d_temp_storage,
             &h_price);
    else
      do_run(h_samples,
             num_timesteps,
             num_paths,
             PayoffCall(K),
             dt,
             S0,
             r,
             sigma,
             d_samples,
             d_paths,
             d_cashflows,
             d_svds,
             d_all_out_of_the_money,
             d_temp_storage,
             &h_price);

    auto end = std::chrono::high_resolution_clock::now();
    const float elapsed_time =
       std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    total_elapsed_time += elapsed_time;
  }

  printf("==============\n");
  printf("GPU Longstaff-Schwartz: %.8lf\n", h_price);

  double price = 0.0;

  if( price_put )
    price = binomial_tree(num_timesteps, PayoffPut(K), dt, S0, r, sigma);
  else
    price = binomial_tree(num_timesteps, PayoffCall(K), dt, S0, r, sigma);

  printf("Binonmial             : %.8lf\n", price);

  if( price_put )
    price = black_scholes_merton_put(T, K, S0, r, sigma);
  else
    price = black_scholes_merton_call(T, K, S0, r, sigma);

  printf("European Price        : %.8lf\n", price);

  printf("==============\n");

  printf("elapsed time for each run         : %.3fms\n", total_elapsed_time / num_runs);
  printf("==============\n");

  // Release memory
  free(h_samples);

  {GpuLock lock;platform.finalize();}
  return 0;
}
