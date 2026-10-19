#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
static int rank_,size_;static int pids_[2];
static void ck(int e){if(e!=MPI_SUCCESS){fprintf(stderr,"MPI_ERROR %d\n",e);MPI_Abort(MPI_COMM_WORLD,e);std::abort();}}
extern "C" int study_mpi_start(){ck(MPI_Init(nullptr,nullptr));ck(MPI_Comm_rank(MPI_COMM_WORLD,&rank_));ck(MPI_Comm_size(MPI_COMM_WORLD,&size_));if(size_!=2)MPI_Abort(MPI_COMM_WORLD,3);int pid=getpid();ck(MPI_Allgather(&pid,1,MPI_INT,pids_,1,MPI_INT,MPI_COMM_WORLD));return rank_;}
extern "C" void study_mpi_finish(){ck(MPI_Finalize());}
extern "C" void study_mpi_barrier(){ck(MPI_Barrier(MPI_COMM_WORLD));}
extern "C" void study_mpi_guard(){int bad=0;if(rank_==0){FILE*f=popen("nvidia-smi -i 5 --query-compute-apps=pid --format=csv,noheader,nounits","r");if(!f)bad=1;else{char line[256];while(fgets(line,sizeof(line),f)){char*end;long p=strtol(line,&end,10);if(end==line|| (p!=pids_[0]&&p!=pids_[1]))bad=1;}if(pclose(f)!=0)bad=1;}}ck(MPI_Bcast(&bad,1,MPI_INT,0,MPI_COMM_WORLD));if(bad){fprintf(stderr,"GPU5_EXTERNAL_USE_DEFER\n");MPI_Abort(MPI_COMM_WORLD,9);}}
extern "C" void study_mpi_lengths(unsigned long n){unsigned long v[2];ck(MPI_Allgather(&n,1,MPI_UNSIGNED_LONG,v,1,MPI_UNSIGNED_LONG,MPI_COMM_WORLD));if(v[0]!=n||v[1]!=n)MPI_Abort(MPI_COMM_WORLD,4);}
extern "C" void study_mpi_scatter(float*recv,int nr,const float*send,int ns){MPI_Request q;MPI_Status s;int peer=1-rank_;ck(MPI_Irecv(recv,nr,MPI_FLOAT,peer,0,MPI_COMM_WORLD,&q));ck(MPI_Send(send,ns,MPI_FLOAT,peer,0,MPI_COMM_WORLD));ck(MPI_Wait(&q,&s));}
extern "C" void study_mpi_allgather(float*recv,int nr,const float*send,int ns){MPI_Status s;int peer=1-rank_;ck(MPI_Sendrecv(send,ns,MPI_FLOAT,peer,0,recv,nr,MPI_FLOAT,peer,0,MPI_COMM_WORLD,&s));}
extern "C" void study_mpi_capture(const float*in,int n,float*out){ck(MPI_Gather(in,n,MPI_FLOAT,out,n,MPI_FLOAT,0,MPI_COMM_WORLD));}
extern "C" double study_mpi_max(double x){double y=0;ck(MPI_Reduce(&x,&y,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD));return y;}
extern "C" void study_mpi_abort(){MPI_Abort(MPI_COMM_WORLD,17);}
