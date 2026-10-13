#include <iris/iris.hpp>
#include <stdexcept>
#include "dslash.h"

double dslash_fn(
    const std::vector<su3_vector> &src, 
    std::vector<su3_vector> &dst,
    const std::vector<su3_matrix> &fat,
    const std::vector<su3_matrix> &lng,
    std::vector<su3_matrix> &fatbck,
    std::vector<su3_matrix> &lngbck,
    size_t *fwd, size_t *bck, size_t *fwd3, size_t *bck3,    
    const size_t iterations,
    size_t wgsize )
{
  int argc=0; char** argv=nullptr;
  iris::Platform platform;
  if(platform.init(&argc,&argv,1)) return 2;
  int count=0,device=-1; iris_device_count(&count);
  for(int i=0;i<count;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    fprintf(stderr,"IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) device=i;
  }
  if(device<0) return 2;

  if(wgsize==0 || wgsize>1024 || even_sites_on_node>2147483647) throw std::runtime_error("invalid launch size");
  const int even_count=static_cast<int>(even_sites_on_node);
  const size_t vector_bytes=sites_on_node*sizeof(su3_vector);
  const size_t matrix_bytes=sites_on_node*4*sizeof(su3_matrix);
  const size_t index_bytes=sites_on_node*4*sizeof(size_t);
  iris::DMem d_src(const_cast<su3_vector*>(src.data()),vector_bytes);
  iris::DMem d_dst(dst.data(),vector_bytes);
  iris::DMem d_fat(const_cast<su3_matrix*>(fat.data()),matrix_bytes);
  iris::DMem d_lng(const_cast<su3_matrix*>(lng.data()),matrix_bytes);
  iris::DMem d_fatbck(fatbck.data(),matrix_bytes),d_lngbck(lngbck.data(),matrix_bytes);
  iris::DMem d_fwd(fwd,index_bytes),d_bck(bck,index_bytes),d_fwd3(fwd3,index_bytes),d_bck3(bck3,index_bytes);
  {
    iris::Task task;
    void* p[]={&d_fat,&d_lng,&d_bck,&d_bck3,&d_fatbck,&d_lngbck,const_cast<int*>(&even_count)};
    int info[]={iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw,sizeof(even_count)};
    size_t g[]={even_sites_on_node},l[]={1};
    if(task.kernel("make_back",1,nullptr,g,l,7,p,info)||task.submit(device,nullptr,1)) throw std::runtime_error("back links failed");
  }
  auto start=Clock::now();
  for(size_t iter=0;iter<iterations+warmups;++iter) {
    if(iter==warmups) start=Clock::now();
    iris::Task task;
    void* p[]={&d_fat,&d_lng,&d_fatbck,&d_lngbck,&d_src,&d_dst,&d_fwd,&d_bck,&d_fwd3,&d_bck3,const_cast<int*>(&even_count)};
    int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,iris_r,iris_r,iris_r,iris_r,sizeof(even_count)};
    size_t g[]={((even_sites_on_node+wgsize-1)/wgsize)*wgsize},l[]={wgsize};
    if(task.kernel("dslash",1,nullptr,g,l,11,p,info)||task.submit(device,nullptr,1)) throw std::runtime_error("dslash failed");
  }
  const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
  iris::Task flush;
  flush.flush_out(d_dst);flush.flush_out(d_fatbck);flush.flush_out(d_lngbck);
  if(flush.submit(device,nullptr,1)) throw std::runtime_error("flush failed");
  platform.finalize();
  return seconds;
}
