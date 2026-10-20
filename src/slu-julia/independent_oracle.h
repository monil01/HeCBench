#pragma once
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <omp.h>
struct SparseOracleResult{bool pass;double componentwise, normwise, solve_backward;unsigned long long product_terms;};
// Independent longdouble full sparse matrix product, including fill-cancellation zeros.
static SparseOracleResult independent_lu(const std::vector<unsigned>&ptr,const std::vector<unsigned>&rows,const std::vector<unsigned>&diag,const std::vector<float>&initial,const std::vector<float>&lu,const std::vector<float>&x,const std::string&matrix){
 const size_t n=diag.size();if(ptr.size()!=n+1||rows.size()!=lu.size()||initial.size()!=lu.size()||x.size()!=n)return{false,0,0,0,0};
 bool finite=true;for(float f:lu)finite &=std::isfinite(f);for(float f:x)finite &=std::isfinite(f);if(!finite)return{false,INFINITY,INFINITY,INFINITY,0};
 const int nt=16;std::vector<std::vector<double>>errrows(nt,std::vector<double>(n,0));std::vector<double>anormrows(n,0);double maxcomponent=0;unsigned long long terms=0;
 for(size_t j=0;j<n;j++)for(unsigned p=ptr[j];p<ptr[j+1];p++)anormrows[rows[p]]+=std::abs(double(initial[p]));
 #pragma omp parallel num_threads(nt) reduction(max:maxcomponent) reduction(+:terms)
 {
  const int tid=omp_get_thread_num();std::vector<long double>sum(n),mag(n),expect(n);std::vector<unsigned>stamp(n,0),touched;touched.reserve(n);
  #pragma omp for schedule(dynamic,8)
  for(size_t col=0;col<n;col++){
   const unsigned tag=unsigned(col)+1;touched.clear();
   auto touch=[&](unsigned r){if(stamp[r]!=tag){stamp[r]=tag;sum[r]=mag[r]=expect[r]=0;touched.push_back(r);}};
   for(unsigned p=ptr[col];p<ptr[col+1];p++){unsigned r=rows[p];touch(r);expect[r]=initial[p];}
   for(unsigned p=ptr[col];p<=diag[col];p++){
    unsigned k=rows[p];long double u=lu[p];if(u==0)continue;touch(k);sum[k]+=u;mag[k]+=std::abs(u);terms++;
    for(unsigned q=diag[k]+1;q<ptr[k+1];q++){
     unsigned r=rows[q];long double term=(long double)lu[q]*u;touch(r);sum[r]+=term;mag[r]+=std::abs(term);terms++;
    }
   }
   for(unsigned r:touched){long double e=std::abs(sum[r]-expect[r]);double rel=double(e/(mag[r]+std::abs(expect[r])+1e-30L));maxcomponent=std::max(maxcomponent,rel);errrows[tid][r]+=double(e);}
  }
 }
 double normA=*std::max_element(anormrows.begin(),anormrows.end()),maxrowerr=0;
 for(size_t i=0;i<n;i++){double e=0;for(auto&r:errrows)e+=r[i];maxrowerr=std::max(maxrowerr,e);}
 // Independent original MatrixMarket parser: original unscaled Ax=1, no source NICSLU transformations.
 std::ifstream file(matrix);if(!file)return{false,maxcomponent,0,INFINITY,terms};std::string line;do{if(!std::getline(file,line))return{false,0,0,INFINITY,terms};}while(line.empty()||line[0]=='%');size_t nr,nc,nnz;std::istringstream h(line);h>>nr>>nc>>nnz;if(nr!=n||nc!=n)return{false,0,0,INFINITY,terms};
 std::vector<long double>ax(n,0),absax(n,0);
 for(size_t p=0;p<nnz;p++){size_t r,c;long double a;if(!(file>>r>>c>>a)||r==0||r>n||c==0||c>n)return{false,0,0,INFINITY,terms};long double t=a*(long double)x[c-1];ax[r-1]+=t;absax[r-1]+=std::abs(t);}
 double backward=0;for(size_t i=0;i<n;i++)backward=std::max(backward,double(std::abs(ax[i]-1.L)/(absax[i]+1.L)));
 double normwise=maxrowerr/std::max(normA,1e-30);return{maxcomponent<=1e-4&&normwise<=1e-4&&backward<=1e-4,maxcomponent,normwise,backward,terms};
}
