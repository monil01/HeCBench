# MNIST convolutional learning with IRIS DMem

Retains all original convolution, pooling, fully-connected forward/backprop
kernels,atomic additions,64x64 launches,layer shapes,dt,threshold,CLI and dataset
loader. Global Layer weight/bias initialization consumes rand before main as in
CUDA; vectors preserve original constructor rand order. DMem binds after IRIS
initialization, source srand123 is restored, and reset buffers call DMem.update.
No native HIP allocations or launches occur in the IRIS host.

Default dataset argument ./data and2epochs retained. Actual recovered DVC IDX
assets are in exclusive local state. Optional HECBENCH_MNIST_TRAIN_LIMIT and
HECBENCH_MNIST_TEST_LIMIT bound real sample counts without changing defaults.
HECBENCH_MNIST_OUTPUT dumps all test logits and every final model weight/bias.
Isolated native adapter adds identical limits/dumps and reconstructs missing
layer/loader headers from CUDA source; original kernel algorithms unchanged.
