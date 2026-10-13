
//#define COMPUTE_COSTS_FULL
//#define COMPUTE_M_SINGLE
//#define COMPUTE_M_ITERATE

const int COSTS_BLOCKSIZE_X = 32;
const int COSTS_BLOCKSIZE_Y = 8;

const int COMPUTE_M_BLOCKSIZE_X = 128; //must be divisible by 2

const int REDUCE_BLOCKSIZE_X = 128;
const int REDUCE_ELEMENTS_PER_THREAD = 8;

const int REMOVE_BLOCKSIZE_X = 32;
const int REMOVE_BLOCKSIZE_Y = 8;

const int UPDATE_BLOCKSIZE_X = 32;
const int UPDATE_BLOCKSIZE_Y = 8;

const int APPROX_SETUP_BLOCKSIZE_X = 32;
const int APPROX_SETUP_BLOCKSIZE_Y = 8;

const int APPROX_M_BLOCKSIZE_X = 128;


