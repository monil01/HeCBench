MCMD retains the shared CUDA host simulation and input defaults. All force
calculations requested by gpu on use the original GPU force kernel through
IRIS dmem tasks. Run from an isolated copy of the dataset to keep simulation
outputs separate. Native HIP thermo and trajectory artifacts are the numeric
baseline. The original host simulation steps are not a device fallback.
