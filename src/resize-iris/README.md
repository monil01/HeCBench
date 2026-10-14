# resize IRIS port status

Native HIP/IRIS validation compares complete output for all six paths.

Set HECBENCH_DUMP to a file prefix to save all six output buffers (u8/u16/u32,
nearest/bilinear). Default CLI, integer ratio arithmetic, eight-channel kernel
groups, zero-initialized remainder channels, and original grid are preserved.
The independent native dump adapter adds diagnostics only; every defined
output is compared, and native algorithms are not repaired.
