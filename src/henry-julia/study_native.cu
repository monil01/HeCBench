#define main legacy_main
#include "henry-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>

double host_rng(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}

// Compute the Boltzmann factor of methane at point (x, y, z) inside structure
//   Loop over all atoms of unit cell of crystal structure
//   Find nearest image to methane at point (x, y, z) for application of periodic boundary conditions
//   Compute energy contribution due to this atom via the Lennard-Jones potential
 double host_compute(double x, double y, double z,
    const StructureAtom * __restrict__ structureAtoms,
    double natoms, double L) 
{
  // (x, y, z) : Cartesian coords of methane molecule
  // structureAtoms : pointer array storing info on unit cell of crystal structure
  // natoms : number of atoms in crystal structure
  // L : box length
  // returns Boltzmann factor e^{-E/(RT)}
  double E = 0.0;  // energy (K)

  // loop over atoms in crystal structure
  for (int i = 0; i < natoms; i++) {
    // Compute distance in each coordinate from (x, y, z) to this structure atom
    double dx = x - structureAtoms[i].x;
    double dy = y - structureAtoms[i].y;
    double dz = z - structureAtoms[i].z;

    // apply nearest image convention for periodic boundary conditions
    const double boxupper = 0.5 * L;
    const double boxlower = -boxupper;

    dx = (dx >  boxupper) ? dx-L : dx;
    dx = (dx >  boxupper) ? dx-L : dx;
    dy = (dy >  boxupper) ? dy-L : dy;
    dy = (dy <= boxlower) ? dy-L : dy;
    dz = (dz <= boxlower) ? dz-L : dz;
    dz = (dz <= boxlower) ? dz-L : dz;

    // compute inverse distance
    double rinv = 1.0 / sqrt(dx*dx + dy*dy + dz*dz);

    // Compute contribution to energy of adsorbate at (x, y, z) due to this atom
    // Lennard-Jones potential (this is the efficient way to compute it)
    double sig_ovr_r = rinv * structureAtoms[i].sigma;
    double sig_ovr_r6 = pow(sig_ovr_r, 6);
    double sig_ovr_r12 = sig_ovr_r6 * sig_ovr_r6;
    E += 4.0 * structureAtoms[i].epsilon * (sig_ovr_r12 - sig_ovr_r6);
  }
  return exp(-E / (R * T));  // return Boltzmann factor
}

int main(){std::ifstream in("/home/imo/HeCBench_julia/src/henry-cuda/IRMOF-1.cssr");std::string line;double L;int na;getline(in,line);L=std::stod(line);getline(in,line);getline(in,line);na=std::stoi(line);getline(in,line);std::map<std::string,double>eps={{"Zn",96.152688},{"O",66.884614},{"C",88.480032},{"H",57.276566}},sig={{"Zn",3.095775},{"O",3.424075},{"C",3.580425},{"H",3.150565}};std::vector<StructureAtom>a(na);for(int i=0;i<na;i++){getline(in,line);std::istringstream s(line);int id;std::string e;double x,y,z;s>>id>>e>>x>>y>>z;a[i]={L*x,L*y,L*z,eps[e],sig[e]};}const int n=4096;std::vector<double>want(n),out(n);for(int i=0;i<n;i++){uint64_t seed=i;double x=L*host_rng(&seed),y=L*host_rng(&seed),z=L*host_rng(&seed);want[i]=host_compute(x,y,z,a.data(),na,L);}StructureAtom*da;double*dy;cudaMalloc(&da,na*sizeof(StructureAtom));cudaMalloc(&dy,n*8);cudaMemcpy(da,a.data(),na*sizeof(StructureAtom),cudaMemcpyHostToDevice);auto run=[&](){insertions<<<16,256>>>(dy,da,na,L);};run();if(cudaDeviceSynchronize()!=cudaSuccess)return 2;for(int trial=1;trial<=5;trial++){double total=0;for(int k=0;k<10;k++){cudaDeviceSynchronize();auto t=std::chrono::steady_clock::now();run();if(cudaDeviceSynchronize()!=cudaSuccess)return 2;total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/10;}cudaMemcpy(out.data(),dy,n*8,cudaMemcpyDeviceToHost);double err=0;bool ok=true;for(int i=0;i<n;i++){err=std::max(err,fabs(out[i]-want[i]));ok=ok&&std::isfinite(out[i])&&fabs(out[i]-want[i])<=1e-10+1e-9*fabs(want[i]);}char path[64];snprintf(path,sizeof(path),"trial_%d.bin",trial);FILE*f=fopen(path,"wb");fwrite(out.data(),8,n,f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.15g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.15g,\"timing_region\":\"mean_insertion_kernel_excluding_host_reduction\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,total,ok?"true":"false",n,err,path,n);if(!ok)return 1;}cudaFree(da);cudaFree(dy);}
