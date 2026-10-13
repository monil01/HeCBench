/** Problem size along one side; total number of cells is this squared */
#ifndef NUM
#define NUM 1024
#endif

// block size
#define BLOCK_SIZE 128

#define Real float
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f

/** SOR relaxation parameter */
const Real omega = 1.85f;
