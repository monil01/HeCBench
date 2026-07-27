# Mojo GPU port of the `bfs` HeCBench benchmark (synthetic-input variant).
#
# Level-synchronous BFS on a deterministic synthetic graph — no file I/O.
# The previous port used string parsing that broke on Mojo 1.0.0b2.
# Two kernels per level (expand, promote); the "any active" flag is
# reduced on the host to avoid needing device atomics.
# Verified against a host BFS from the same source node.

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from std.memory import UnsafePointer
from std.sys import argv, exit
from std.time import perf_counter_ns


comptime N_NODES: Int = 4096
comptime EDGES_PER_NODE: Int = 8


def bfs_expand(
        starting: UnsafePointer[Int32, MutAnyOrigin],
        counts:   UnsafePointer[Int32, MutAnyOrigin],
        edges:    UnsafePointer[Int32, MutAnyOrigin],
        graph_mask: UnsafePointer[Int8, MutAnyOrigin],
        upmask:     UnsafePointer[Int8, MutAnyOrigin],
        visited:    UnsafePointer[Int8, MutAnyOrigin],
        cost:       UnsafePointer[Int32, MutAnyOrigin],
        n: Int32):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(n):
        return
    if graph_mask[tid] == Int8(0):
        return
    graph_mask[tid] = Int8(0)
    var s = Int(starting[tid])
    var c = Int(counts[tid])
    for k in range(s, s + c):
        var id_ = Int(edges[k])
        if visited[id_] == Int8(0):
            cost[id_] = cost[tid] + Int32(1)
            upmask[id_] = Int8(1)


def bfs_promote(
        graph_mask: UnsafePointer[Int8, MutAnyOrigin],
        upmask:     UnsafePointer[Int8, MutAnyOrigin],
        visited:    UnsafePointer[Int8, MutAnyOrigin],
        n: Int32):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(n): return
    if upmask[tid] == Int8(1):
        graph_mask[tid] = Int8(1)
        visited[tid] = Int8(1)
        upmask[tid] = Int8(0)


def main() raises:
    print("Synthetic graph: nodes =", N_NODES, ", ~edges/node =", EDGES_PER_NODE)

    var ctx = DeviceContext()
    var d_start = ctx.enqueue_create_buffer[DType.int32](N_NODES)
    var d_count = ctx.enqueue_create_buffer[DType.int32](N_NODES)

    # Build the graph deterministically. Each node gets EDGES_PER_NODE
    # neighbours (with repeats allowed), from an LCG.
    var elen = N_NODES * EDGES_PER_NODE
    var d_edges = ctx.enqueue_create_buffer[DType.int32](elen)
    var s: UInt64 = 20260721
    with d_start.map_to_host() as sh, d_count.map_to_host() as ch,\
         d_edges.map_to_host() as eh:
        for i in range(N_NODES):
            sh[i] = Int32(i * EDGES_PER_NODE)
            ch[i] = Int32(EDGES_PER_NODE)
        for k in range(elen):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            eh[k] = Int32(Int(s >> 33) % N_NODES)

    var d_mask   = ctx.enqueue_create_buffer[DType.int8](N_NODES)
    var d_upmask = ctx.enqueue_create_buffer[DType.int8](N_NODES)
    var d_visit  = ctx.enqueue_create_buffer[DType.int8](N_NODES)
    var d_cost   = ctx.enqueue_create_buffer[DType.int32](N_NODES)
    with d_mask.map_to_host() as mh, d_upmask.map_to_host() as uh,\
         d_visit.map_to_host() as vh, d_cost.map_to_host() as ch:
        for i in range(N_NODES):
            mh[i] = Int8(0); uh[i] = Int8(0); vh[i] = Int8(0)
            ch[i] = Int32(-1)
        mh[0] = Int8(1); vh[0] = Int8(1); ch[0] = Int32(0)

    comptime BLOCK: Int = 256
    var blocks = (N_NODES + BLOCK - 1) // BLOCK

    ctx.synchronize()
    var t0 = perf_counter_ns()
    var levels = 0
    while True:
        ctx.enqueue_function[func=bfs_expand](
            d_start.unsafe_ptr(), d_count.unsafe_ptr(), d_edges.unsafe_ptr(),
            d_mask.unsafe_ptr(), d_upmask.unsafe_ptr(), d_visit.unsafe_ptr(),
            d_cost.unsafe_ptr(), Int32(N_NODES),
            grid_dim=blocks, block_dim=BLOCK)
        ctx.enqueue_function[func=bfs_promote](
            d_mask.unsafe_ptr(), d_upmask.unsafe_ptr(), d_visit.unsafe_ptr(),
            Int32(N_NODES), grid_dim=blocks, block_dim=BLOCK)
        ctx.synchronize()
        var any_active = False
        with d_mask.map_to_host() as mh:
            for i in range(N_NODES):
                if mh[i] == Int8(1):
                    any_active = True; break
        levels = levels + 1
        if not any_active: break
    var us = Float64(perf_counter_ns() - t0) / 1e3
    print("Total kernel execution time:", us, "(us) over", levels, "levels")

    # Host reference BFS
    var ref_cost = ctx.enqueue_create_buffer[DType.int32](N_NODES)
    with d_start.map_to_host() as sh, d_count.map_to_host() as ch,\
         d_edges.map_to_host() as eh, ref_cost.map_to_host() as rh:
        for i in range(N_NODES): rh[i] = Int32(-1)
        rh[0] = Int32(0)
        var frontier = List[Int]()
        frontier.append(0)
        while len(frontier) > 0:
            var nextf = List[Int]()
            for k in range(len(frontier)):
                var u = frontier[k]
                var s2 = Int(sh[u])
                var c = Int(ch[u])
                for e in range(s2, s2 + c):
                    var vv = Int(eh[e])
                    if rh[vv] == Int32(-1):
                        rh[vv] = rh[u] + Int32(1)
                        nextf.append(vv)
            frontier = nextf.copy()

    var mismatches = 0
    with d_cost.map_to_host() as gh, ref_cost.map_to_host() as rh:
        for i in range(N_NODES):
            if gh[i] != rh[i]:
                mismatches = mismatches + 1
    if mismatches == 0:
        print("PASS")
    else:
        print("FAIL (", mismatches, "mismatches)")
