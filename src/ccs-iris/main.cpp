/*
 Condition-dependent Correlation Subgroups (CCS)
 Description: Biclustering has been emerged as a powerful tool for
 identification of a group of co-expressed genes under a subset
 of experimental conditions (measurements) present in a gene
 expression dataset.  In this program we implemented CCS biclustering.

 Developer: Dr. Anindya Bhattacharya and Dr. Yan Cui, UTHSC, Memphis, TN, USA
 Email: anindyamail123@gmail.com; ycui2@uthsc.edu

 Note: The minimum number of genes and the samples per bicluster is 10.
 User can alter the minimum size by changing the values for 'mingene'
 and 'minsample' defined in "ccs.h" file for minimum number of genes and samples
 respectively.
*/

#include <chrono>
#include <iris/iris.hpp>
#include <vector>
#include "ccs.h"
#include "matrixsize.c"
#include "readgene.c"
#include "pair_cor.c"
#include "bicluster_pair_score.c"
#include "merge_bicluster.c"
#include "print_bicluster.c"

int main(int argc, char *argv[])
{
  FILE *in,*out;
  struct gn *gene;
  char **Hd;
  char *infile,*outfile;
  int c, errflag;
  int maxbcn=MAXB;
  int print_type=0;
  int repeat=0;
  int i,n,D;
  extern char *optarg;
  float thr;
  struct bicl *bicluster;
  float overlap=100.f;

  infile = outfile = NULL;
  in = out = NULL;

  errflag = n = D = 0;
  thr = 0.f;

  while ((c = getopt(argc, argv, "ht:m:r:i:p:o:g:?")) != -1)
  {
    switch(c)
    {
      case 'h': // help
        printUsage();
        exit(0);
      case 't': // threshold value
        thr = atof(optarg);
        break;
      case 'm': // maximum number of bicluster search
        maxbcn = atoi(optarg);
        break;
      case 'r': // kernel repeat times
        repeat = atoi(optarg);
        break;
      case 'g': // output file format
        overlap = atof(optarg);
        break;
      case 'p': // output file format
        print_type = atoi(optarg);
        break;
      case 'i': // the input expression file
        infile = optarg;
        break;
      case 'o': // the output file
        outfile = optarg;
        break;
      case ':':       /* -f or -o without operand */
        printf("Option -%c requires an operand\n", optopt);
        errflag++;
        break;
      case '?':
        fprintf(stderr,"Unrecognized option: -%c\n", optopt);
        errflag++;
    }
  }

  if (thr == 0)
  {
    fprintf(stderr,"***** WARNING: Threshold Theta (corr coeff) "
        "value assumed to be ZERO (0)\n");
  }

  if (outfile == NULL)
  {
    fprintf(stderr,"***** WARNING: Output file assumed to be STDOUT\n");
    out = stdout;
  }
  else if ((out = fopen(outfile,"w")) == NULL) //write open bicluster file
  {
    fprintf(stderr,"***** ERROR: Unable to open Output file %s\n",outfile);
    errflag++;
  }

  if ((thr < 0) || (thr > 1))
  {
    fprintf(stderr,"***** ERROR: Threshold Theta (corr coeff) "
        "must be between 0.0-1.0\n");
  }

  if (infile == NULL)
  {
    fprintf(stderr,"***** ERROR: Input file not defined\n");
    if (out) fclose(out);
    errflag++;
  }
  else if ((in = fopen(infile,"r")) == NULL)  //open gene file
  {
    fprintf(stderr,"***** ERROR: Unable to open Input %s\n", infile);
    if (out) fclose(out);
    errflag++;
  }

  if (errflag)
  {
    printUsage();
    exit(1);
  }

  getmatrixsize(in,&n,&D);
  printf("Number of rows=%d\tNumber of columns=%d\n",n,D);

  if(maxbcn>n) maxbcn=n;

  gene = (struct gn *)calloc(n,sizeof(struct gn));
  Hd = (char **)calloc(D+1,sizeof(char *));

  for (i = 0; i < n; i++)
    gene[i].x = ( float *)calloc(D+1,sizeof( float));

  bicluster = (struct bicl *)calloc(maxbcn,sizeof(struct bicl));
  for (i = 0; i < maxbcn; i++)
  {
    bicluster[i].sample = (char *)calloc(D,sizeof(char));
    bicluster[i].data = (char *)calloc(n,sizeof(char));
  }

  // initialize the gene data
  readgene(infile,gene,Hd,n,D);

  auto start = std::chrono::steady_clock::now();

  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  std::vector<float> flat_gene(n*(D+1));
  for (i=0; i<n; ++i) memcpy(flat_gene.data()+i*(D+1), gene[i].x, sizeof(float)*(D+1));
  std::vector<float> scores(maxbcn);
  std::vector<int> counts(maxbcn), samplecounts(maxbcn);
  std::vector<char> samples(D*maxbcn), data(n*maxbcn), tmpsamples(D*maxbcn), tmpdata(n*maxbcn);
  iris::DMem mg(flat_gene.data(), flat_gene.size()*sizeof(float));
  iris::DMem ms(scores.data(), scores.size()*sizeof(float));
  iris::DMem mc(counts.data(), counts.size()*sizeof(int));
  iris::DMem msc(samplecounts.data(), samplecounts.size()*sizeof(int));
  iris::DMem msa(samples.data(), samples.size());
  iris::DMem md(data.data(), data.size());
  iris::DMem mts(tmpsamples.data(), tmpsamples.size());
  iris::DMem mtd(tmpdata.data(), tmpdata.size());
  auto kstart = std::chrono::steady_clock::now();
  for (i=0; i<repeat; ++i) {
    iris::Task task;
    void* params[]={&mg,&n,&maxbcn,&D,&thr,&msa,&md,&ms,&mc,&msc,&mts,&mtd};
    int info[]={iris_r,sizeof(n),sizeof(maxbcn),sizeof(D),sizeof(thr),iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw};
    size_t global[]={static_cast<size_t>(maxbcn)}, local[]={1};
    task.kernel("compute_bicluster",1,nullptr,global,local,12,params,info);
    if(i==repeat-1) {
      task.flush_out(ms); task.flush_out(mc); task.flush_out(msc);
      task.flush_out(mts); task.flush_out(mtd);
    }
    task.submit(iris_gpu,nullptr,1);
  }
  auto kend = std::chrono::steady_clock::now();
  printf("Average kernel execution time %f (s)\n",std::chrono::duration<double>(kend-kstart).count()/repeat);
  float *bicluster_temp_score = scores.data();
  int *bicluster_temp_datacount = counts.data();
  int *bicluster_temp_samplecount = samplecounts.data();
  for(i=0; i<maxbcn; ++i) {
    memcpy(bicluster[i].sample,tmpsamples.data()+D*i,D);
    memcpy(bicluster[i].data,tmpdata.data()+n*i,n);
  }
  platform.finalize();

  for(i=0; i<maxbcn; i++) {
    bicluster[i].score=bicluster_temp_score[i];
    bicluster[i].datacount=bicluster_temp_datacount[i];
    bicluster[i].samplecount=bicluster_temp_samplecount[i];
  }

  printbicluster(out,gene,Hd,n,D,maxbcn,thr,bicluster,print_type,overlap);

  for (i = 0; i < n; i++) {
    free(gene[i].x);
    free(gene[i].id);
  }
  free(gene);
  for (i = 0; i < D+1; i++) free(Hd[i]);
  free(Hd);

  for (i = 0; i < maxbcn; i++) {
    free(bicluster[i].sample);
    free(bicluster[i].data);
  }

  free(bicluster);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Elapsed time = %f (s)\n", time * 1e-9f);
  if (print_type==0) fprintf(out,"\n\nElapsed time = %f s\n", time * 1e-9f);
  if (out) fclose(out);

  return 0;
}
