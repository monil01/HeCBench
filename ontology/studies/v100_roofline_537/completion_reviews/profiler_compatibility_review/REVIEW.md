# V100 / Ubuntu24.04 profiler compatibility — targeted unprivileged option

**Investigation only: no profiler binary run, GPU query, download, extraction,
installation, system package/OS mutation, compiler/build, benchmark or tests.**
Active2022 collectors/captures remain untouched. Parent owns any future experiment.

## Conclusion, with exact scope

As of this official-source lookup (21October2026), **Nsight Compute2025.2.1** is the
latest listed patch in the last release family whose archived support page still
includes Volta/GV100. Official2025.3 removes Volta. The Ubuntu24.04 x86_64 NVIDIA
repository contains the exact2025.2.1.3 package, and bundled2025.2 system requirements
explicitly discuss Ubuntu24.04 and require Linux glibc>=2.27. This host is Ubuntu
24.04.5/x86_64/glibc2.39. Hence2025.2 is an applicable host/GPU candidate rather than
an OS-upgrade proposal. This is **not** a demonstrated fix for miniDGS.

**An alternative is already installed:**
`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Compute/ncu`.
Its bundled official release notes identify2025.2.0; static ELF strings show
`2025.2.0.0 (build 35613519) (public-release)`. Its GV100 table says supported.
The x86_64 wrapper resolves its own sibling
`target/linux-desktop-glibc_2_11_3-x64/ncu` (62MiB). These are static-file findings;
`--version` was not executed and dynamic load/profiling success is unproved.
The `/2025/` HPC path is an alias of the same bundle, not an independent version.
The ordinary `/usr/bin/ncu` resolves `/usr/lib/nsight-compute/ncu` (existing2022
campaign stratum), so do not replace that executable or change a global PATH.

## Primary official evidence

1. [Archived2025.2 landing/support page](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_2)
   includes GV100, names2025.2Update1, and recommends Linux575.57.08 or newer.
2. [Archived2025.3 landing page](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3)
   lists supported GPUs starting at Turing. The
   [release history](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)
   explicitly states Volta support was dropped in2025.3.
3. [CUDA12.9Update1 component/driver table](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)
   identifies Nsight Compute2025.2.1.3 and Linux driver575.57.08. It explains newer
   drivers retain CUDA compatibility. Read-only `/proc/driver/nvidia/version` is
   **580.178.04**, above the Nsight2025.2 recommended driver. Compatibility on this
   exact MPI/libsystemd stack remains an empirical question; a newer-driver number
   is not an unconditional injection-library compatibility guarantee.
4. [Official Ubuntu24.04/x86_64 package index](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/)
   lists `nsight-compute-2025.2.1_2025.2.1.3-1_amd64.deb`, label283MB,
   2025-06-04. It also lists2025.2.0.11 and later2025.3+ packages; those newer major
   versions do not restore Volta. Repository placement is additional primary
   evidence for the intended Ubuntu24.04 packaging, not a successful runtime test.
5. Bundled official docs (SHA-bound in local_inventory.json):
   `.../docs/ReleaseNotes/topics/{gpu-support,platform-support,system-requirements}.html`.
   The requirements say Linux glibc2.27 minimum and distinguish optional GUI/Qt
   dependencies; the CLI proposal does not request GUI packages. Public archived
   docs links redirected to archive.docs.nvidia.com and the web reader could not
   retrieve them; that access limitation is recorded rather than silently claiming
   a retrieved version-specific webpage. Local version-bound official copies and
   official public landing/package tables provide the support evidence.

The observation that580 satisfies the recommended>=575 driver range is an inference
from those official requirements plus the proc file, not an actual profiling run.

## Preferred minimal option: use existing2025.2.0 side by side

No download or extraction is needed to investigate the installed executable.
Root may first explicitly review its exact ELF/wrapper/injection/performance-library
and section hashes, then a CPU `--version`/`--help` check under owned private paths.
Only after root authorizes a new GPU6-locked calibration phase should any GPU work
run. No active manager-bound tool needs to be overwritten for investigation: use a
new reviewed collector/preflight adapter selecting this absolute executable.
Do not change existing2022 source tools, caches, capture directories or global env.

### Optional patch-release acquisition — NOT performed

Exact official user-space payload URL:

`https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-compute-2025.2.1_2025.2.1.3-1_amd64.deb`

The repository label is **283MB**; exact byte size, package hash and Installed-Size
were **not retrieved** (web reader rejected the8.15MB Packages index size). Do not
present a rounded label or guessed SHA as verified package bytes. Future root-
authorized acquisition should first bind NVIDIA's repository package metadata and
verify the complete payload SHA/length before private extraction. No driver,
toolkit, apt repository or OS dependency changes are proposed.

After future authorization and successful verification, the extraction operation
is ordinary unprivileged archive extraction, **not package installation**:

```
dpkg-deb --extract /tmp/OWNED_NEW_DOWNLOAD/nsight-compute-2025.2.1_2025.2.1.3-1_amd64.deb \
  /tmp/OWNED_NEW_NCU2025_2_1_PREFIX
```

This command was not run. No package-manager install/upgrade, maintainer-script
execution, driver update, root filesystem writes or privilege escalation is needed.
The executable should then be addressed by exact extracted path after inspecting
the archive layout; do not assume a guessed path exists or replace `/usr/bin/ncu`.
If required CLI libraries are absent, record a blocker and consider separately
reviewed user-space dependencies; do not invoke system mutation.

## Storage budget

Existing bundle logical file sum1,326,108,365B; unique-inode file content1,228,121,397B.
These are **local2025.2.0** inventory sizes, not measured2025.2.1 extraction size or
filesystem allocated blocks. Already installed use adds no payload/extraction disk.
At inventory time `/tmp` free12,204,781,568B, home2,764,046,336B; ongoing capture
activity can change those values. Prefer private `/tmp`, never home or old captures.
For a future .deb route, conservatively budget **at least4GiB** before acquisition
(rounded compressed payload + version-neighbor expanded size + staging margin +
minimum2GiB campaign reserve), and inspect exact Installed-Size before extraction.
This is an operational estimate, not a measured peak; calibrations/reports need an
additional explicit bounded allowance. Recheck storage and competing worker budget
before every new artifact. Do not clean/delete unrelated evidence to create space.

## Required fresh method/calibration stratum before any benchmark recapture

- Archive and bind chosen profiler executable/wrapper, target injection libraries,
  NvPerf libraries/sections, driver/host info, CLI help/version and tool/guard SHA.
  Keep2022captures and2025captures independent; do not silently relabel their tools.
- Under GPU6-only common lock, private TMPDIR and existing child write guard, query
  only that GPU's metric availability. Confirm every requested DRAM/L2 bytes,
  duration, pred-on add/mul/FMA, integer/tensor and peak ceiling metric name, unit,
  availability and aggregation. Missing counters stay null/unknown; no substitution
  by a similarly named metric without explicit reviewed definitions.
- Fresh source-known arithmetic calibration for FP32/FP64/packedFP16 lane factors
  and FMA=2 convention. Existing2022half/tensor calibration is not automatically a
 2025proof. If tensor conversion is needed, repeat precise WMMA/HMMA instruction
  convention calibration for the new tool; unsupported precision never fabricated.
- Revalidate first-function/geometry selector behavior, kernel vs application replay,
  private lock path, source before/after/PCI identity, complete raw CSV parsing and
  stopping behavior with the actual new version. Preserve clock-control none and
  cache-control all policy if comparable; name any altered instrumentation strata.
- MPI workaround experiment should preserve original miniDGS2rank/sameGPU6 numeric
  workflow, both actual process outputs, original five checks and source ROI. New
  profiler/process filters must not drop a rank or any checker. A small source-
  justified control precedes costly recapture, but cannot certify full original
  geometry. Host-safe_close EBADF under2022and the queued040filter remain preserved.
- Only a successful complete new control can support the claim this profiler solves
  that observed method failure. A failure is a separate attempt, not proof that the
  benchmark/framework is unsupported. Counter samples still do not imply complete
  primary ROI, valid elapsed benchmark timing or causal bottleneck.
- Do not merge2022/2025counter values, tool-dependent ceilings, or calibration factors
  into one homogeneous dataset without explicit version covariates and reviewed
  comparability controls. Original primary five-run timings never replaced.

## Scope of the current failure

Parent reports native miniDGS023 NCU2022.4.1 host libsystemd safe_close/EBADF and
040process-filter recovery queued. This review establishes a plausible supported
alternate **environment/method**, not the unique cause of that assertion or its fix.
No OS upgrade is warranted by this evidence. The new binaries need not replace
or interrupt any active/root-owned controller. All future execution remains root's
separate decision after this report; current work makes no GPU run requests.
