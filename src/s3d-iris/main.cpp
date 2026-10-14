#include <iris/iris.hpp>
#include "gpu_lock.hpp"
#include <vector>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include "OptionParser.h"
#include "Utility.h"

using namespace std;

// Forward Declarations
void addBenchmarkSpecOptions(OptionParser &op);
void RunBenchmark(OptionParser &op);

// ****************************************************************************
// Function: main
//
// Purpose:
//   The main function takes care of initialization (device and MPI),  then
//   performs the benchmark and prints results.
//
// Arguments:
//
//
// Programmer: Jeremy Meredith
// Creation:
//
// Modifications:
//   Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//   Split timing reports into detailed and summary.  For serial code, we
//   report all trial values, and for parallel, skip the per-process vals.
//   Also detect and print outliers from parallel runs.
//
// ****************************************************************************
int main(int argc, char *argv[])
{
    int ret = 0;
    iris::Platform platform;{GpuLock lock;platform.init(&argc,&argv,1);}
    int devices=0;iris_device_count(&devices);for(int d=0;d<devices;d++){int b=0;size_t z=0;iris_device_info(d,iris_backend,&b,&z);printf("IRIS backend=%d HIP=%d\n",b,iris_hip);}

    try
    {

        // Get args
        OptionParser op;

        //Add shared options to the parser
        op.addOption("device", OPT_VECINT, "0",
                "specify device(s) to run on", 'd');
        op.addOption("verbose", OPT_BOOL, "", "enable verbose output", 'v');
        op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');
        op.addOption("size", OPT_INT, "1", "specify problem size", 's');
        op.addOption("quiet", OPT_BOOL, "", "write minimum necessary to standard output", 'q');

        addBenchmarkSpecOptions(op);

        if (!op.parse(argc, argv))
        {
            op.usage();
            return (op.HelpRequested() ? 0 : 1);
        }

        // Run the benchmark
        RunBenchmark(op);
    }
    catch( std::exception& e )
    {
        std::cerr << e.what() << std::endl;
        ret = 1;
    }
    catch( ... )
    {
        ret = 1;
    }

    {GpuLock lock;platform.finalize();}
    return ret;
}

template<class real> void RunTest(OptionParser& op){
 int sp[]={24,32,40,48},dp[]={16,24,32,40};int size=op.getOptionInt("size")-1;
 if(size<0||size>3)throw std::runtime_error("size class");int edge=sizeof(real)==8?dp[size]:sp[size],n=edge*edge*edge;
 int passes=op.getOptionInt("passes");if(passes<1)throw std::runtime_error("passes");
 std::string type=sizeof(real)==8?"double":"float";
 std::vector<real> t(n,1000),p(n,1.0132e6),y(size_t(22)*n),wdot(size_t(22)*n),molwt(22,1),
 rf(size_t(206)*n),rb(size_t(206)*n),rklow(size_t(21)*n),c(size_t(22)*n),a(size_t(121)*n),eg(size_t(32)*n);
 for(int j=0;j<22;j++)for(int i=0;i<n;i++)y[size_t(j)*n+i]=j==14?0.064:j==3?0.218:j==21?0.718:0;
 real tconv=1,pconv=1,rateconv=1;
 auto start=std::chrono::steady_clock::now();
 {
 GpuLock lock;
 iris::DMem gpu_t(t.data(),t.size()*sizeof(real)),gpu_p(p.data(),p.size()*sizeof(real)),gpu_y(y.data(),y.size()*sizeof(real)),gpu_wdot(wdot.data(),wdot.size()*sizeof(real)),gpu_molwt(molwt.data(),molwt.size()*sizeof(real)),gpu_rf(rf.data(),rf.size()*sizeof(real)),gpu_rb(rb.data(),rb.size()*sizeof(real)),gpu_rklow(rklow.data(),rklow.size()*sizeof(real)),gpu_c(c.data(),c.size()*sizeof(real)),gpu_a(a.data(),a.size()*sizeof(real)),gpu_eg(eg.data(),eg.size()*sizeof(real));
 for(int i=0;i<passes;i++){{iris::Task task;void* p[]={&gpu_t,&gpu_rf,&tconv};int info[]={iris_r,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("ratt_kernel_"+type).c_str(),1,nullptr,&global,&local,3,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_eg,&tconv};int info[]={iris_r,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("rdsmh_kernel_"+type).c_str(),1,nullptr,&global,&local,3,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_p,&gpu_t,&gpu_y,&gpu_c,&tconv,&pconv};int info[]={iris_r,iris_r,iris_r,iris_rw,sizeof(real),sizeof(real)};size_t local=128,global=n;task.kernel(("gr_base_"+type).c_str(),1,nullptr,&global,&local,6,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_rf,&gpu_rb,&gpu_eg,&tconv};int info[]={iris_r,iris_rw,iris_rw,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("ratt2_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_rf,&gpu_rb,&gpu_eg,&tconv};int info[]={iris_r,iris_rw,iris_rw,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("ratt3_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_rf,&gpu_rb,&gpu_eg,&tconv};int info[]={iris_r,iris_rw,iris_rw,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("ratt4_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_rf,&gpu_rb,&gpu_eg,&tconv};int info[]={iris_r,iris_rw,iris_rw,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("ratt5_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_rf,&gpu_rb,&gpu_eg,&tconv};int info[]={iris_r,iris_rw,iris_rw,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("ratt6_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_rf,&gpu_rb,&gpu_eg,&tconv};int info[]={iris_r,iris_rw,iris_rw,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("ratt7_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_rf,&gpu_rb,&gpu_eg,&tconv};int info[]={iris_r,iris_rw,iris_rw,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("ratt8_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_rf,&gpu_rb,&gpu_eg,&tconv};int info[]={iris_r,iris_rw,iris_rw,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("ratt9_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_rklow,&tconv};int info[]={iris_r,iris_rw,sizeof(real)};size_t local=128,global=n;task.kernel(("ratt10_kernel_"+type).c_str(),1,nullptr,&global,&local,3,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_c,&gpu_rf,&gpu_rb,&gpu_rklow,&tconv};int info[]={iris_r,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(real)};size_t local=64,global=n;task.kernel(("ratx_kernel_"+type).c_str(),1,nullptr,&global,&local,6,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_t,&gpu_c,&gpu_rf,&gpu_rb,&gpu_rklow,&tconv};int info[]={iris_r,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(real)};size_t local=64,global=n;task.kernel(("ratxb_kernel_"+type).c_str(),1,nullptr,&global,&local,6,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_c,&gpu_rf,&gpu_rb};int info[]={iris_rw,iris_rw,iris_rw};size_t local=128,global=n;task.kernel(("ratx2_kernel_"+type).c_str(),1,nullptr,&global,&local,3,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_c,&gpu_rf,&gpu_rb};int info[]={iris_rw,iris_rw,iris_rw};size_t local=128,global=n;task.kernel(("ratx4_kernel_"+type).c_str(),1,nullptr,&global,&local,3,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_a};int info[]={iris_rw,iris_rw,iris_rw};size_t local=128,global=n;task.kernel(("qssa_kernel_"+type).c_str(),1,nullptr,&global,&local,3,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_a};int info[]={iris_rw,iris_rw,iris_rw};size_t local=64,global=n;task.kernel(("qssab_kernel_"+type).c_str(),1,nullptr,&global,&local,3,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_a};int info[]={iris_rw,iris_rw,iris_rw};size_t local=128,global=n;task.kernel(("qssa2_kernel_"+type).c_str(),1,nullptr,&global,&local,3,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_wdot,&rateconv,&gpu_molwt};int info[]={iris_rw,iris_rw,iris_rw,sizeof(real),iris_r};size_t local=128,global=n;task.kernel(("rdwdot_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_wdot,&rateconv,&gpu_molwt};int info[]={iris_rw,iris_rw,iris_rw,sizeof(real),iris_r};size_t local=128,global=n;task.kernel(("rdwdot2_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_wdot,&rateconv,&gpu_molwt};int info[]={iris_rw,iris_rw,iris_rw,sizeof(real),iris_r};size_t local=128,global=n;task.kernel(("rdwdot3_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_wdot,&rateconv,&gpu_molwt};int info[]={iris_rw,iris_rw,iris_rw,sizeof(real),iris_r};size_t local=128,global=n;task.kernel(("rdwdot6_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_wdot,&rateconv,&gpu_molwt};int info[]={iris_rw,iris_rw,iris_rw,sizeof(real),iris_r};size_t local=128,global=n;task.kernel(("rdwdot7_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_wdot,&rateconv,&gpu_molwt};int info[]={iris_rw,iris_rw,iris_rw,sizeof(real),iris_r};size_t local=128,global=n;task.kernel(("rdwdot8_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_wdot,&rateconv,&gpu_molwt};int info[]={iris_rw,iris_rw,iris_rw,sizeof(real),iris_r};size_t local=128,global=n;task.kernel(("rdwdot9_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}
{iris::Task task;void* p[]={&gpu_rf,&gpu_rb,&gpu_wdot,&rateconv,&gpu_molwt};int info[]={iris_rw,iris_rw,iris_rw,sizeof(real),iris_r};size_t local=128,global=n;task.kernel(("rdwdot10_kernel_"+type).c_str(),1,nullptr,&global,&local,5,p,info);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("S3D task failed");}}
 iris::Task flush;flush.flush_out(gpu_wdot);flush.submit(iris_gpu,nullptr,1);
 }
 printf("\nAverage time of executing s3d kernels: %lf (us)\n",std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/passes);
 for(int i=0;i<22;i++){printf("% 23.16E ",double(wdot[size_t(i)*n]));if(i%3==2)printf("\n");}printf("\n");
 for(real v:wdot)if(!std::isfinite(v))throw std::runtime_error("nonfinite S3D output");
 if(const char* path=getenv("S3D_DUMP_DIR")){
  std::string file=std::string(path)+"/"+type+".bin";FILE*fp=fopen(file.c_str(),"wb");if(!fp)throw std::runtime_error("dump open");fwrite(wdot.data(),sizeof(real),wdot.size(),fp);fclose(fp);
 }
}
void addBenchmarkSpecOptions(OptionParser&){}
void RunBenchmark(OptionParser& op){RunTest<float>(op);RunTest<double>(op);}
