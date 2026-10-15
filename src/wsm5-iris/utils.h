#define XXX 8
#define YYY 8
#define TODEV(A,s) A=(float*)malloc((s)*sizeof(float)); \
 for(int index=0;index<(s);++index) A[index]=0.001; \
 iris::DMem A##_d(A,(s)*sizeof(float))
#define FROMDEV(A,s) if(A##_d.fetch(A)!=IRIS_SUCCESS) return 1
#define FREE(A) free(A)
#define TODEV3(A) TODEV(A,d3)
#define TODEV2(A) TODEV(A,d2)
#define FROMDEV3(A) FROMDEV(A,d3)
#define FROMDEV2(A) FROMDEV(A,d2)
