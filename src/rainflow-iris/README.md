IRIS DMem rainflow retains seed123, scale100 history-length distribution,
num_history/repeat arguments, default100000 100,256threads and original extra-grid
rounding. CUDA/CPU algorithms and exact full result-length memcmp are preserved.
Shared RFResult is three doubles(count,range,mean), replacing backend-specific
vector spelling with identical24-byte POD layout. Workspace/output buffers iris_rw;
history inputs explicitly update and lengths flush before CPU check. Original
verification checks lengths only; full cycle triples are computed but not compared.
CUDA kernel source unvalidated; timing includes IRIS task overhead.
