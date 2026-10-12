# degrid IRIS port

IRIS `DMem` port of the degrid benchmark. The host preserves the native
initialization and CPU reference check; the HIP backend exposes the degrid
kernel as `degrid_kernel_double` and applies the original image/GCF pointer
offsets inside the kernel.
