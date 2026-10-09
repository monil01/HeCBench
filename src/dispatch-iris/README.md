# dispatch IRIS

This port maps the CUDA/HIP empty-kernel launch timing loops to IRIS task
submissions. Asynchronous task submissions are followed by `iris_synchronize()`
where the native HIP benchmark synchronizes with the device.
