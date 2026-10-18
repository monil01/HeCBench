using CUDA,JACC,Printf
JACC.@init_backend
include("../study_generated.jl")
function sg_jacc_bb_bin_histo_4(i,j,a_d_bin_counter,a_d_seg_begins,a_d_seg_ends,a_num_segs,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_bb_bin_histo(ctx,CSpan(a_d_bin_counter,Int32(0)),CSpan(a_d_seg_begins,Int32(0)),CSpan(a_d_seg_ends,Int32(0)),a_num_segs)
 nothing
end
function launch_generated(::Val{:bb_bin_histo_4},a_d_bin_counter,a_d_seg_begins,a_d_seg_ends,a_num_segs;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_bb_bin_histo_4,a_d_bin_counter,a_d_seg_begins,a_d_seg_ends,a_num_segs,Int32(threads))
 nothing
end
function sg_jacc_bb_bin_group_5(i,j,a_d_bin_segs_id,a_d_bin_counter,a_d_seg_begins,a_d_seg_ends,a_num_segs,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_bb_bin_group(ctx,CSpan(a_d_bin_segs_id,Int32(0)),CSpan(a_d_bin_counter,Int32(0)),CSpan(a_d_seg_begins,Int32(0)),CSpan(a_d_seg_ends,Int32(0)),a_num_segs)
 nothing
end
function launch_generated(::Val{:bb_bin_group_5},a_d_bin_segs_id,a_d_bin_counter,a_d_seg_begins,a_d_seg_ends,a_num_segs;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_bb_bin_group_5,a_d_bin_segs_id,a_d_bin_counter,a_d_seg_begins,a_d_seg_ends,a_num_segs,Int32(threads))
 nothing
end
function sg_jacc_gen_copy_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_copy(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_copy_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_copy_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk256_wp2_tc1_r2_r2_orig_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk256_wp2_tc1_r2_r2_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk256_wp2_tc1_r2_r2_orig_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk256_wp2_tc1_r2_r2_orig_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_wp2_tc2_r3_r4_orig_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_wp2_tc2_r3_r4_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_wp2_tc2_r3_r4_orig_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_wp2_tc2_r3_r4_orig_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_wp2_tc4_r5_r8_orig_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_wp2_tc4_r5_r8_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_wp2_tc4_r5_r8_orig_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_wp2_tc4_r5_r8_orig_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_wp4_tc4_r9_r16_strd_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_wp4_tc4_r9_r16_strd(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_wp4_tc4_r9_r16_strd_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_wp4_tc4_r9_r16_strd_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_wp8_tc4_r17_r32_strd_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_wp8_tc4_r17_r32_strd(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_wp8_tc4_r17_r32_strd_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_wp8_tc4_r17_r32_strd_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_wp16_tc4_r33_r64_strd_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_wp16_tc4_r33_r64_strd(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_wp16_tc4_r33_r64_strd_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_wp16_tc4_r33_r64_strd_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk256_wp8_tc16_r65_r128_strd_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk256_wp8_tc16_r65_r128_strd(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk256_wp8_tc16_r65_r128_strd_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk256_wp8_tc16_r65_r128_strd_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk256_wp32_tc8_r129_r256_strd_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk256_wp32_tc8_r129_r256_strd(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk256_wp32_tc8_r129_r256_strd_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk256_wp32_tc8_r129_r256_strd_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_tc4_r257_r512_orig_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_tc4_r257_r512_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_tc4_r257_r512_orig_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_tc4_r257_r512_orig_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk256_tc4_r513_r1024_orig_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk256_tc4_r513_r1024_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk256_tc4_r513_r1024_orig_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk256_tc4_r513_r1024_orig_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk512_tc4_r1025_r2048_orig_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk512_tc4_r1025_r2048_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk512_tc4_r1025_r2048_orig_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk512_tc4_r1025_r2048_orig_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_copy_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_copy(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_copy_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_copy_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk256_wp2_tc1_r2_r2_orig_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk256_wp2_tc1_r2_r2_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk256_wp2_tc1_r2_r2_orig_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk256_wp2_tc1_r2_r2_orig_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_wp2_tc2_r3_r4_orig_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_wp2_tc2_r3_r4_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_wp2_tc2_r3_r4_orig_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_wp2_tc2_r3_r4_orig_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_wp2_tc4_r5_r8_orig_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_wp2_tc4_r5_r8_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_wp2_tc4_r5_r8_orig_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_wp2_tc4_r5_r8_orig_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_wp4_tc4_r9_r16_strd_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_wp4_tc4_r9_r16_strd(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_wp4_tc4_r9_r16_strd_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_wp4_tc4_r9_r16_strd_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_wp8_tc4_r17_r32_strd_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_wp8_tc4_r17_r32_strd(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_wp8_tc4_r17_r32_strd_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_wp8_tc4_r17_r32_strd_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_wp16_tc4_r33_r64_strd_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_wp16_tc4_r33_r64_strd(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_wp16_tc4_r33_r64_strd_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_wp16_tc4_r33_r64_strd_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk256_wp8_tc16_r65_r128_strd_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk256_wp8_tc16_r65_r128_strd(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk256_wp8_tc16_r65_r128_strd_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk256_wp8_tc16_r65_r128_strd_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk256_wp32_tc8_r129_r256_strd_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk256_wp32_tc8_r129_r256_strd(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk256_wp32_tc8_r129_r256_strd_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk256_wp32_tc8_r129_r256_strd_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk128_tc4_r257_r512_orig_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk128_tc4_r257_r512_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk128_tc4_r257_r512_orig_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk128_tc4_r257_r512_orig_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk256_tc4_r513_r1024_orig_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk256_tc4_r513_r1024_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk256_tc4_r513_r1024_orig_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk256_tc4_r513_r1024_orig_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_gen_bk512_tc4_r1025_r2048_orig_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_gen_bk512_tc4_r1025_r2048_orig(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_bin_size)
 nothing
end
function launch_generated(::Val{:gen_bk512_tc4_r1025_r2048_orig_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_gen_bk512_tc4_r1025_r2048_orig_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_bin_size,Int32(threads))
 nothing
end
function sg_jacc_kern_block_sort_6(i,j,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_kern_block_sort(ctx,CSpan(a_key,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_workloads_per_block)
 nothing
end
function launch_generated(::Val{:kern_block_sort_6},a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_kern_block_sort_6,a_key,a_keyB,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block,Int32(threads))
 nothing
end
function sg_jacc_kern_block_merge_7(i,j,a_keys,a_keysB,a_seg_begins,a_seg_ends,a_bin,a_stride,a_workloads_per_block,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_kern_block_merge(ctx,CSpan(a_keys,Int32(0)),CSpan(a_keysB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_stride,a_workloads_per_block)
 nothing
end
function launch_generated(::Val{:kern_block_merge_7},a_keys,a_keysB,a_seg_begins,a_seg_ends,a_bin,a_stride,a_workloads_per_block;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_kern_block_merge_7,a_keys,a_keysB,a_seg_begins,a_seg_ends,a_bin,a_stride,a_workloads_per_block,Int32(threads))
 nothing
end
function sg_jacc_kern_copy_6(i,j,a_srck,a_dstk,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_kern_copy(ctx,CSpan(a_srck,Int32(0)),CSpan(a_dstk,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_workloads_per_block)
 nothing
end
function launch_generated(::Val{:kern_copy_6},a_srck,a_dstk,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_kern_copy_6,a_srck,a_dstk,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block,Int32(threads))
 nothing
end
function sg_jacc_kern_block_sort_8(i,j,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_kern_block_sort(ctx,CSpan(a_key,Int32(0)),CSpan(a_val,Int32(0)),CSpan(a_keyB,Int32(0)),CSpan(a_valB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_workloads_per_block)
 nothing
end
function launch_generated(::Val{:kern_block_sort_8},a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_kern_block_sort_8,a_key,a_val,a_keyB,a_valB,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block,Int32(threads))
 nothing
end
function sg_jacc_kern_block_merge_9(i,j,a_keys,a_vals,a_keysB,a_valsB,a_seg_begins,a_seg_ends,a_bin,a_stride,a_workloads_per_block,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_kern_block_merge(ctx,CSpan(a_keys,Int32(0)),CSpan(a_vals,Int32(0)),CSpan(a_keysB,Int32(0)),CSpan(a_valsB,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_stride,a_workloads_per_block)
 nothing
end
function launch_generated(::Val{:kern_block_merge_9},a_keys,a_vals,a_keysB,a_valsB,a_seg_begins,a_seg_ends,a_bin,a_stride,a_workloads_per_block;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_kern_block_merge_9,a_keys,a_vals,a_keysB,a_valsB,a_seg_begins,a_seg_ends,a_bin,a_stride,a_workloads_per_block,Int32(threads))
 nothing
end
function sg_jacc_kern_copy_8(i,j,a_srck,a_srcv,a_dstk,a_dstv,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block,nt)
 ctx=(Int32((i-1)%nt),Int32((i-1)÷nt),Int32(j-1),nt)
 sg_kern_copy(ctx,CSpan(a_srck,Int32(0)),CSpan(a_srcv,Int32(0)),CSpan(a_dstk,Int32(0)),CSpan(a_dstv,Int32(0)),CSpan(a_seg_begins,Int32(0)),CSpan(a_seg_ends,Int32(0)),CSpan(a_bin,Int32(0)),a_workloads_per_block)
 nothing
end
function launch_generated(::Val{:kern_copy_8},a_srck,a_srcv,a_dstk,a_dstv,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block;blocks,threads)
 gx,gy=blocks isa Tuple ? blocks : (blocks,1)
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(gx,gy),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(gx*threads,gy),sg_jacc_kern_copy_8,a_srck,a_srcv,a_dstk,a_dstv,a_seg_begins,a_seg_ends,a_bin,a_workloads_per_block,Int32(threads))
 nothing
end
include("../study_common.jl")
segsort_study(launch_generated)
