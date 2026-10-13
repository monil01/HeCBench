#include <iris/iris.hpp>
#include <chrono>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include "utils.h"
#include "validation.h"
#include "validation_lease.h"

static void checked(int rc) { if(rc!=IRIS_SUCCESS) throw std::runtime_error("IRIS deredundancy task failed"); }
static void run(const std::string& name,size_t global,size_t local,std::initializer_list<void*> args,
                std::initializer_list<int> modes,std::initializer_list<iris::DMem*> outputs={}) {
 std::vector<void*> a(args); std::vector<int> info(modes);
 if(a.size()!=info.size()) throw std::runtime_error("kernel argument mismatch");
 size_t gws[]={global},lws[]={local}; iris::Task task;
 checked(task.kernel(name.c_str(),1,nullptr,gws,lws,a.size(),a.data(),info.data()));
 for(auto* m:outputs) checked(task.flush_out(*m));
 checked(task.submit(iris_gpu,nullptr,1));
}
struct Buffers {
 iris::DMem lengths,offsets,reads,packed,gaps,indices,orders,words,magic,cutoff,cluster,table,representative;
 Buffers(std::vector<int>& l,std::vector<long>& o,std::vector<char>& r,std::vector<unsigned int>& p,
         std::vector<int>& g,std::vector<unsigned short>& idx,std::vector<unsigned short>& ord,
         std::vector<long>& w,std::vector<int>& m,std::vector<int>& cut,std::vector<int>& cl,
         std::vector<unsigned short>& t,int& rep)
 : lengths(l.data(),l.size()*sizeof(int)),offsets(o.data(),o.size()*sizeof(long)),reads(r.data(),r.size()),
 packed(p.data(),p.size()*sizeof(unsigned int)),gaps(g.data(),g.size()*sizeof(int)),
 indices(idx.data(),idx.size()*sizeof(unsigned short)),orders(ord.data(),ord.size()*sizeof(unsigned short)),
 words(w.data(),w.size()*sizeof(long)),magic(m.data(),m.size()*sizeof(int)),cutoff(cut.data(),cut.size()*sizeof(int)),
 cluster(cl.data(),cl.size()*sizeof(int)),table(t.data(),t.size()*sizeof(unsigned short)),representative(&rep,sizeof(rep)) {}
};
int main(int argc,char** argv) {
 Option option; checkOption(argc,argv,option);
 std::vector<Read> records; if(readFile(records,option)) return 1;
 int n=records.size(); std::vector<int> lengths(n); std::vector<long> offsets(n+1);
 for(int i=0;i<n;++i) {lengths[i]=records[i].data.size();offsets[i+1]=offsets[i]+lengths[i]/16*16+16;}
 long total=offsets[n];
 std::vector<char> bases(total);
 for(int i=0;i<n;++i) memcpy(bases.data()+offsets[i],records[i].data.data(),lengths[i]);
 std::vector<unsigned int> packed(total/16); std::vector<int> gaps(n),magic(4*n),cutoff(n),clusters(n,-1);
 std::vector<unsigned short> indices(total),orders(total),table(65536,0); std::vector<long> words(n);
 int representative=-1,wordLength=option.wordLength; float threshold=option.threshold;
 size_t row_work=size_t((n+127)/128)*128;
 iris::Platform platform; std::unique_ptr<Buffers> memory;
 auto t1=std::chrono::high_resolution_clock::now();
 {
   ValidationGPULease lease("DEREDUNDANCY_GPU_LEASE");
   checked(platform.init(nullptr,nullptr,1)); bool hip=false;
   for(int d=0;d<platform.ndevs();++d) {
     int backend=0,type=0;char name[256]={};size_t bytes=0;
     checked(iris_device_info(d,iris_backend,&backend,&bytes)); checked(iris_device_info(d,iris_type,&type,&bytes)); checked(iris_device_info(d,iris_name,name,&bytes));
     printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name);hip|=backend==3&&type==256;
   }
   if(!hip) return 1;
   memory.reset(new Buffers(lengths,offsets,bases,packed,gaps,indices,orders,words,magic,cutoff,clusters,table,representative)); auto& b=*memory;
   run("kernel_baseToNumber",128*128,128,{&b.reads,&total},{iris_rw,sizeof(total)});
   run("kernel_compressData",row_work,128,{&b.lengths,&b.offsets,&b.reads,&b.packed,&b.gaps,&n},{iris_r,iris_r,iris_r,iris_rw,iris_w,sizeof(n)});
   run("kernel_createIndex"+std::to_string(wordLength),row_work,128,{&b.reads,&b.lengths,&b.offsets,&b.indices,&b.orders,&b.words,&b.magic,&n},
       {iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_w,iris_w,sizeof(n)});
   run("kernel_createCutoff",row_work,128,{&threshold,&wordLength,&b.lengths,&b.words,&b.cutoff,&n},{sizeof(threshold),sizeof(wordLength),iris_r,iris_r,iris_w,sizeof(n)},
       {&b.indices,&b.offsets,&b.words});
 } // CPU sorting does not hold the GPU lease.
 for(int i=0;i<n;++i) std::sort(indices.data()+offsets[i],indices.data()+offsets[i]+words[i]);
 {
   std::unique_ptr<ValidationGPULease> lease(new ValidationGPULease("DEREDUNDANCY_GPU_LEASE")); auto& b=*memory;
   b.indices.update(indices.data());
   run("kernel_mergeIndex",row_work,128,{&b.offsets,&b.indices,&b.orders,&b.words,&n},{iris_r,iris_r,iris_rw,iris_r,sizeof(n)});
   int completed=0;
   auto lease_start=std::chrono::steady_clock::now();
   auto phase_start=lease_start;
   while(representative<n) {
     run("kernel_updateRepresentative",1,1,{&b.cluster,&b.representative,&n},{iris_rw,iris_rw,sizeof(n)},{&b.representative});
     if(representative>=n-1) break;
     if(getenv("DEREDUNDANCY_GPU_LEASE")) {std::cout<<"CLUSTER_PROGRESS representative="<<representative<<" / "<<n<<std::endl;}
     run("kernel_makeTable",128*128,128,{&b.offsets,&b.indices,&b.orders,&b.words,&b.table,&representative},{iris_r,iris_r,iris_r,iris_r,iris_rw,sizeof(representative)});
     run("kernel_magic",row_work,128,{&threshold,&b.lengths,&b.magic,&b.cluster,&representative,&n},{sizeof(threshold),iris_r,iris_r,iris_rw,sizeof(representative),sizeof(n)});
     run("kernel_filter",size_t(n)*128,128,{&threshold,&wordLength,&b.lengths,&b.offsets,&b.indices,&b.orders,&b.words,&b.cutoff,&b.cluster,&b.table,&n},
         {sizeof(threshold),sizeof(wordLength),iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,iris_r,sizeof(n)});
     run("kernel_align",row_work,128,{&threshold,&b.lengths,&b.offsets,&b.packed,&b.gaps,&representative,&b.cluster,&n},
         {sizeof(threshold),iris_r,iris_r,iris_r,iris_r,sizeof(representative),iris_rw,sizeof(n)});
     run("kernel_cleanTable",128*128,128,{&b.offsets,&b.indices,&b.orders,&b.words,&b.table,&representative},{iris_r,iris_r,iris_r,iris_r,iris_rw,sizeof(representative)});
     ++completed;
     if(getenv("DEREDUNDANCY_GPU_LEASE") &&
        (completed%16==0 || std::chrono::duration<double>(std::chrono::steady_clock::now()-lease_start).count()>20)) {
       lease.reset();
       if(std::chrono::duration<double>(std::chrono::steady_clock::now()-phase_start).count()>1800) throw std::runtime_error("validation clustering phase budget1800s");
       lease.reset(new ValidationGPULease("DEREDUNDANCY_GPU_LEASE")); lease_start=std::chrono::steady_clock::now();
     }
   }
   iris::Task flush; checked(flush.flush_out(b.cluster));
   if(getenv("DEREDUNDANCY_DUMP")) for(auto* m:{&b.reads,&b.packed,&b.gaps,&b.indices,&b.orders,&b.words,&b.magic,&b.cutoff,&b.table}) checked(flush.flush_out(*m));
   checked(flush.submit(iris_gpu,nullptr,1));
   memory.reset();int errors=platform.error_count();checked(platform.finalize());if(errors) return 1;
 }
 auto t2=std::chrono::high_resolution_clock::now();
 printf("Device offload time %lf secs \n",std::chrono::duration<double>(t2-t1).count());
 deredundancy_dump(lengths.data(),offsets.data(),words.data(),bases.data(),indices.data(),orders.data(),packed.data(),gaps.data(),magic.data(),cutoff.data(),clusters.data(),table.data(),n);
 std::ofstream file(option.outputFile.c_str());int sum=0;
 for(int i=0;i<n;++i) if(clusters[i]==i) {file<<records[i].name<<std::endl;file<<records[i].data<<std::endl;++sum;}
 file.close();std::cout<<"cluster count: "<<sum<<std::endl;
 return 0;
}
