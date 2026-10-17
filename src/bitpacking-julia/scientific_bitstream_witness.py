"""Independent complete bitstream decoder and strided reduction witness."""
import array
import json
import pathlib
import sys
SIZES=(2,123,3411,83621,872163,1049603)
result=json.loads(pathlib.Path(sys.argv[1]).read_text());rows=[]
for model,record in result['models'].items():
    for trial in record['trials']:
        data=array.array('I');data.frombytes(pathlib.Path(trial['output_artifact']['saved_path']).read_bytes());offset=0;checked=0
        for n in SIZES:
            grid=min(4096,(n+255)//256);words=((13*n)//64+1)*2;minimum,width=data[offset:offset+2];offset+=2
            assert minimum==87231 and width==13
            lows=data[offset:offset+grid];offset+=grid;highs=data[offset:offset+grid];offset+=grid;packed=data[offset:offset+words];offset+=words
            low=[2**31-1]*grid;high=[-2**31]*grid;raw=packed.tobytes()
            for i in range(n):
                expected=87231+(73*i+19*(i//257))%8192
                if i==0:expected=87231
                if i==1:expected=95422
                group=(i//256)%grid;low[group]=min(low[group],expected);high[group]=max(high[group],expected)
                startbit=i*13;byte=startbit//8;shift=startbit%8
                decoded=(int.from_bytes(raw[byte:byte+3],'little')>>shift)&8191
                assert decoded+minimum==expected
                checked+=1
            assert list(lows)==low and list(highs)==high
            used=n*13
            assert all(((raw[bit//8]>>(bit%8))&1)==0 for bit in range(used,len(raw)*8))
        assert offset==len(data)
        rows.append({'model':model,'trial':trial['trial'],'all_decoded_input_values':checked,'all_minimum_width_scanpartials_and_padding_correct':True})
assert len(rows)==20
print(json.dumps({'benchmark':'bitpacking','trials':rows,'independent_method':'Decode every 13-bit element via little-endian byte-window extraction and compare independently generated input; separate strided min/max witness and zero padding check. No GPU reduction/shared tile algorithm reused.','scope':'Positive13bitrangeInt32 six named sizes,notzerowidth/allintegerdatatypes/100million'},indent=2))
