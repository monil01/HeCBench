"""Generate the frozen UInt64 fixture; GPU results are never oracle inputs."""
from pathlib import Path
import struct
mask=(1<<64)-1
values=[0,mask,1<<63,1,(1<<63)-1,0x5555555555555555,0xaaaaaaaaaaaaaaaa]
values += [1<<j for j in range(64)] + [mask^(1<<j) for j in range(64)]
values += [(v<<(8*k))&mask for k in range(8) for v in range(256)]
state=2
while len(values)<4099:
    state=(state*6364136223846793005+1442695040888963407)&mask
    values.append(state)
Path(__file__).with_name('study_input.bin').write_bytes(struct.pack('<4099Q',*values))
