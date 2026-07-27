#!/usr/bin/env bash
# HeCBench environment for the zenith machine (RTX 5090 + HPC SDK 26.3).
#
# Usage:  source tools/env.sh
# Sets:
#   - PATH   → HPC SDK compilers (nvcc, nvc++, nvc) + CUDA 13.1
#   - CUDA_ARCH   sm_120       (Blackwell)
#   - HECBENCH_SM cc120        (nvc++ -gpu= form)
#   - HECBENCH_ROOT             absolute repo path

# Resolve the repo root regardless of how the script is invoked
if [[ -n "${BASH_SOURCE[0]}" ]]; then
  __hecbench_script="${BASH_SOURCE[0]}"
else
  __hecbench_script="${0}"
fi
export HECBENCH_ROOT="$(cd -- "$(dirname -- "$__hecbench_script")/.." && pwd)"

# NVIDIA HPC SDK 26.3
HPC_SDK=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3
if [[ -d "$HPC_SDK" ]]; then
  export PATH="$HPC_SDK/compilers/bin:$HPC_SDK/cuda/bin:$PATH"
  # cudarc / other runtime dl-loaders look for libcudart.so, libnvrtc.so, etc.
  # HPC SDK 26.3 tucks these under cuda/13.1/targets/x86_64-linux/lib.
  _HPC_CUDA_LIB="$HPC_SDK/cuda/13.1/targets/x86_64-linux/lib"
  export LD_LIBRARY_PATH="$HPC_SDK/compilers/lib:$HPC_SDK/cuda/lib64:$_HPC_CUDA_LIB:${LD_LIBRARY_PATH:-}"
  unset _HPC_CUDA_LIB
else
  echo "warning: HPC SDK not found at $HPC_SDK" >&2
fi

# Rust (rustup, user-space)
[[ -d "$HOME/.cargo/bin"  ]] && export PATH="$HOME/.cargo/bin:$PATH"

# Julia (juliaup, user-space)
[[ -d "$HOME/.juliaup/bin" ]] && export PATH="$HOME/.juliaup/bin:$PATH"

# pixi (user-space package manager — needed for Mojo)
[[ -d "$HOME/.pixi/bin"    ]] && export PATH="$HOME/.pixi/bin:$PATH"

# Mojo via the pixi env at src/_mojo_env
export MOJO_ENV="$HECBENCH_ROOT/src/_mojo_env"
if [[ -d "$MOJO_ENV/.pixi/envs/default/bin" ]]; then
  export PATH="$MOJO_ENV/.pixi/envs/default/bin:$PATH"
  # Mojo 1.0.0b2 requires MODULAR_HOME to find its stdlib config; without
  # this, `from std.gpu.host import DeviceContext` fails with "unable to
  # locate module 'std'".
  export MODULAR_HOME="$MOJO_ENV/.pixi/envs/default/share/max"
fi

# GPU-target settings for this box (RTX 5090 = Blackwell = sm_120)
export CUDA_ARCH=sm_120
export HECBENCH_SM=cc120

# Convenience: build wrappers understood by HeCBench Makefiles
#   make ARCH=$CUDA_ARCH         # for -cuda Makefile
#   make -f Makefile.nvc SM=$HECBENCH_SM   # for -omp with nvc++
export ARCH="$CUDA_ARCH"
export SM="$HECBENCH_SM"

# Sanity check
if command -v nvcc >/dev/null && command -v nvc++ >/dev/null; then
  echo "hecbench: env ready  (nvcc=$(nvcc --version | grep -oE 'release [^,]+' | head -1),"\
       "nvc++=$(nvc++ --version 2>&1 | grep -oE 'nvc\+\+ [0-9.-]+' | head -1),"\
       "arch=$CUDA_ARCH)"
else
  echo "hecbench: env sourced but nvcc/nvc++ not found; check HPC SDK path" >&2
fi

unset __hecbench_script
