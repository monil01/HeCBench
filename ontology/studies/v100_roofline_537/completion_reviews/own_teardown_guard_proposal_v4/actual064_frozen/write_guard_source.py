#!/usr/bin/env python3
"""Unprivileged child-only Landlock write confinement for profiler jobs.

Reads/execution retain existing permissions. Filesystem writes are limited to
explicit fresh output directories, explicitly named existing Julia caches, and
existing character devices. This never grants permissions or changes the OS.
Unsupported restriction support is fatal; there is no unrestricted fallback.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import stat

LIBC=ctypes.CDLL(None,use_errno=True)
WRITE=1<<1
WRITE_ACTIONS=WRITE|sum(1<<n for n in range(4,15))


class Ruleset(ctypes.Structure):
    _fields_=[('handled_access_fs',ctypes.c_uint64)]


class Beneath(ctypes.Structure):
    _pack_=1
    _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]


def checked(value,what):
    if value<0:
        code=ctypes.get_errno()
        raise OSError(code,what+': '+os.strerror(code))
    return value


def capability():
    if platform.machine()!='x86_64':
        raise RuntimeError('This audited syscall binding requires Linux x86_64')
    return checked(LIBC.syscall(444,ctypes.c_void_p(0),0,1),'Landlock ABI query')


def character_devices():
    # Driver/NVML opens may use O_RDWR even for device queries. These existing
    # nodes remain subject to normal user permissions and the fixed GPU binding.
    paths=[Path('/dev/null'),Path('/dev/urandom'),Path('/dev/random')]
    paths.extend(Path('/dev').glob('nvidia*'))
    caps=Path('/dev/nvidia-caps')
    if caps.is_dir():paths.extend(caps.iterdir())
    return [p for p in paths if p.exists() and stat.S_ISCHR(p.stat().st_mode)]


def restrict_writes(directories):
    abi=capability()
    if abi<3:
        raise RuntimeError('Landlock ABI>=3 required to restrict truncation')
    attribute=Ruleset(WRITE_ACTIONS)
    rules=checked(LIBC.syscall(444,ctypes.byref(attribute),ctypes.sizeof(attribute),0),'Create write restriction')
    try:
        for path in directories:
            p=Path(path).resolve()
            if not p.is_dir():raise RuntimeError('Allowed write directory must exist: '+str(p))
            fd=os.open(p,os.O_PATH|os.O_CLOEXEC)
            try:
                rule=Beneath(WRITE_ACTIONS,fd)
                checked(LIBC.syscall(445,rules,1,ctypes.byref(rule),0),'Add allowed write directory '+str(p))
            finally:os.close(fd)
        for p in character_devices():
            fd=os.open(p,os.O_PATH|os.O_CLOEXEC)
            try:
                rule=Beneath(WRITE,fd)
                checked(LIBC.syscall(445,rules,1,ctypes.byref(rule),0),'Add existing character device '+str(p))
            finally:os.close(fd)
        # libcuda names newly created worker threads through
        # /proc/self/task/<tid>/comm. Child PIDs/TIDs do not exist yet when rules
        # are created. Permit existing proc virtual-file writes/truncation only;
        # normal user permissions still apply, with no proc create/remove rights.
        fd=os.open('/proc',os.O_PATH|os.O_CLOEXEC)
        try:
            rule=Beneath(WRITE|(1<<14),fd)
            checked(LIBC.syscall(445,rules,1,ctypes.byref(rule),0),'Allow existing proc virtual-file updates')
        finally:os.close(fd)
        # This only removes the ability to gain privileges in the child.
        checked(LIBC.prctl(38,1,0,0,0),'Set no_new_privs')
        checked(LIBC.syscall(446,rules,0),'Enforce child write restriction')
    finally:os.close(rules)
    return abi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-dir',action='append',required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command
    if command and command[0]=='--':command=command[1:]
    if not command:p.error('A target command is required')
    abi=restrict_writes(a.allow_dir)
    os.write(2,('V100_WRITE_GUARD '+json.dumps({'landlock_abi':abi,
        'write_directories':a.allow_dir,'privilege':'existing unprivileged user; no_new_privs'})+'\n').encode())
    os.execvpe(command[0],command,os.environ)


if __name__=='__main__':main()
