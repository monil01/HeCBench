/* Diagnostic-only link wrapper; the original HPL solver source is untouched. */
#include "hpl.h"
void __real_HPL_pdgesv(HPL_T_grid*,HPL_T_palg*,HPL_T_pmat*);
void __wrap_HPL_pdgesv(HPL_T_grid* grid,HPL_T_palg* algo,HPL_T_pmat* a){
 static int instance=0;__real_HPL_pdgesv(grid,algo,a);
 const char* prefix=getenv("HECBENCH_HPL_DUMP");if(!prefix)return;
 int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 char path[4096];snprintf(path,sizeof(path),"%s.rank%d.case%d.bin",prefix,rank,instance++);
 FILE* f=fopen(path,"wb");if(!f)MPI_Abort(MPI_COMM_WORLD,1);
 int shape[]={a->n,a->nb,a->mp,a->nq,a->ld,a->info,rank};
 fwrite(shape,sizeof(int),7,f);
 for(int j=0;j<a->nq;++j)fwrite(a->A+(size_t)j*a->ld,sizeof(double),a->mp,f);
 fwrite(a->X,sizeof(double),a->nq-1,f);fclose(f);
 printf("HPL_WHOLE_OUTPUT rank=%d N=%d NB=%d active_factors=%zu solution=%d info=%d\n",rank,a->n,a->nb,(size_t)a->mp*a->nq,a->nq-1,a->info);
}
