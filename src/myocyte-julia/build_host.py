from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
out=Path("/home/imo/HeCBench_julia/.porting-state/portable_julia/runtime/myocyte")
out.mkdir(parents=True,exist_ok=True)
for name in ["solver","oracle"]:
 subprocess.run(["g++","-O3","-std=c++17","-fPIC","-shared",str(p/f"study_{name}.cpp"),"-o",str(out/f"libstudy_{name}.so")],check=True)
