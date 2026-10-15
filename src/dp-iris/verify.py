import sys,re,math
from pathlib import Path
a=Path(sys.argv[1]).read_text();b=Path(sys.argv[2]).read_text()
def vals(t):return [(int(x),float(y),float(z)) for x,y,z in re.findall(r"VALUE (\d+) ([^\s]+) ([^\s]+)",t)]
x,y=vals(a),vals(b)
assert len(x)==len(y)==6 and [v[0] for v in x]==[4,4,4,8,8,8],(x,y)
assert len(re.findall(r"(?m)^PASS$",a))==len(re.findall(r"(?m)^PASS$",b))==6 and "FAIL" not in a and "FAIL" not in b
refs=[float(v.split()[-1]) for v in re.findall(r"Independent integer sum [^\n]+",b)]
assert len(refs)==6
for i,(u,v) in enumerate(zip(x,y)):
 assert u==v and math.isfinite(v[1]) and v[1]==v[2]==refs[i],(i,u,v,refs[i])
print(f"PASS all6 native/IRIS outputs and original verdicts EXACT; both types / all3 distinct GPU paths; independent int64 dot={refs[0]}")
