from pathlib import Path
import sys,array,importlib.util
P=Path(__file__).resolve().parent.parent;s=importlib.util.spec_from_file_location('oracle',P/'generate_assets.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
out=Path(sys.argv[1])if len(sys.argv)>1 else P/'wrong_layout_defined_f64.bin';out.write_bytes(array.array('d',o.zero_padded_source64_prefix()).tobytes());print(out)
