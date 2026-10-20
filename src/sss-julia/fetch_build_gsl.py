#!/usr/bin/env python3
"""Isolated user-prefix GSL2.8 dependency reproduction, no system package edits."""
import pathlib,argparse,hashlib,urllib.request,tarfile,subprocess,json
R=pathlib.Path(__file__).resolve().parents[2]
a=argparse.ArgumentParser();a.add_argument('--prefix',type=pathlib.Path,default=R/'.porting-state/portable_julia/sss_readiness/gsl-prefix');a.add_argument('--cache',type=pathlib.Path,default=R/'.porting-state/portable_julia/sss_readiness');z=a.parse_args();prefix=z.prefix.resolve();cache=z.cache.resolve();allowed=(R,pathlib.Path.home().resolve(),pathlib.Path('/tmp'));assert any(prefix.is_relative_to(q)for q in allowed)and any(cache.is_relative_to(q)for q in allowed),'Dependencywritesmustremaininexplicituser/workspace/tmpdirectories';cache.mkdir(parents=True,exist_ok=True)
# Never rebuild over evidence-bound installed archives.
existing=[prefix/'lib'/n for n in ('libgsl.a','libgslcblas.a')]
if any(p.exists()for p in existing):
 assert all(p.is_file()for p in existing),'Partial existing prefix; choose a new empty prefix'
 audit=json.loads((R/'src/sss-julia/final_saved_full_audit.json').read_text())
 expected={pathlib.Path(x['path']).name:x['expected_sha256']for x in audit['source_bindings']if pathlib.Path(x['path']).name in ('libgsl.a','libgslcblas.a')}
 for f in existing:
  with f.open('rb')as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected[f.name],'Existing prefix differs; refuse overwrite, choose a new prefix and explicit toolchain stratum'
 print('Existing evidence-bound GSL archives match; no rebuilding, installation or mutation performed.')
 raise SystemExit(0)
URL='https://ftp.gnu.org/gnu/gsl/gsl-2.8.tar.gz';EXPECTED='6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190';archive=cache/'gsl-2.8.tar.gz'
if not archive.exists():
 with urllib.request.urlopen(URL,timeout=60)as remote,archive.open('xb')as out:
  while chunk:=remote.read(1024*1024):out.write(chunk)
with archive.open('rb')as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==EXPECTED and archive.stat().st_size==8997136,'GSL source identity mismatch'
source=cache/'gsl-2.8'
if not source.exists():
 with tarfile.open(archive)as t:t.extractall(cache,filter='data')
commands=[[str(source/'configure'),'--prefix='+str(prefix),'--disable-shared','--enable-static','CFLAGS=-O3 -fPIC'],['make','-j8'],['make','install']]
for i,command in enumerate(commands):
 with(cache/f'cold_gsl_command{i}.log').open('w')as log:subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
(cache/'cold_gsl_reproduction.json').write_text(json.dumps({'source_URL':URL,'source_sha256':actual,'source_bytes':archive.stat().st_size,'prefix':str(prefix),'commands':commands,'scope':'Writesonlyexplicitusercache/source/prefix; unprivilegedsourcebuild, nosystempackagemanager/OSmutation'},indent=2)+'\n')
