from pathlib import Path
p=Path.cwd()
coords=[(40, 90), (31, 90), (30, 92), (30, 90), (33, 94), (29, 90)]
lines=[]
for i,(lat,lon)in enumerate(coords):
 s=f"ID{i:04d}".ljust(28)+f"{lat:5.1f}{lon:5.1f}"+"X"*10
 assert len(s)==48
 lines.append(s+"\n")
(p/"db.txt").write_text("".join(lines))
(p/"filelist.txt").write_text("db.txt\n")
