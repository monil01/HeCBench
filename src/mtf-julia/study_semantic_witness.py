"""Independent safe witnesses, not replacements for original executable/native correctness."""
import json
def online_mtf(word):
 alphabet=list(range(256));out=[]
 for w in word:
  ix=alphabet.index(w);out.append(ix);alphabet.pop(ix);alphabet.insert(0,w)
 return out
def source_with_initialized_permutation(word):
 # Explicit SECONDARY init-only patch model: remaining exacttwo-loop operations preserved.
 alphabet=list(range(256));out=list(word)
 for w in word:
  ix=alphabet.index(w)
  if alphabet[0]!=w:alphabet[1:ix+1]=alphabet[:ix];alphabet[0]=w
 for counter,symbol in enumerate(alphabet):
  start=0
  while True:
   try:ix=out.index(symbol,start)
   except ValueError:break
   out[ix]=counter;start=ix+1
 return out
word=list(b"ABA");safe_padding=[0]*256+[85];snapshot=safe_padding[:256];safe_padding[1:257]=snapshot;safe_padding[0]=65
result={"first_symbol":65,"original_alphabet_length":256,"original_zero_initialized_find_result":256,"copy_count":256,"destination_first":1,"destination_last":256,"last_valid_index":255,"guard_before":85,"guard_after":safe_padding[256],"source_bug":"Original CPU and GPU copies write one byte beyond declared alphabet on first nonzero input.","input_ASCII":"ABA","standard_online_MTF_indices":online_mtf(word),"init_only_patched_source_global_relabel":source_with_initialized_permutation(word),"semantic_distinction":"Source emits global final-dictionary relabels after scanning allinput, not eachsymbolrankbeforeitsmove. Inplace replacement may alias lateralphabetvalues, so not even a generally immutable map.","interpretation":"Padding witness is safe diagnostic emulation, not nativecorrectness. Init-only model is an explicit hypothetical patch, not a claimed originalrun or validMTF repair."}
assert result['guard_after']==0 and result['standard_online_MTF_indices']==[65,66,1] and result['init_only_patched_source_global_relabel']!=[65,66,1]
print(json.dumps(result,indent=2))
