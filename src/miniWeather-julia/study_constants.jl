
const PI_D = 3.14159265358979323846264338327
const GRAV = 9.8
const CP = 1004.0
const CV = 717.0
const RD = 287.0
const P0 = 1.0e5
const C0 = 27.5629410929725921310572974482
const GAMM = 1.40027894002789400278940027894
const XLEN = 2.0e4
const ZLEN = 1.0e4
const HV_BETA = 0.25
const CFL = 1.50
const MAX_SPEED = 450.0
const HS = Int32(2)
const NUM_VARS = Int32(4)
const ID_DENS = Int32(0)
const ID_UMOM = Int32(1)
const ID_WMOM = Int32(2)
const ID_RHOT = Int32(3)
const DIR_X = Int32(1)
const DIR_Z = Int32(2)
const DATA_SPEC_THERMAL = Int32(2)
const NQPOINTS = 3
const QPOINTS = (0.112701665379258311482073460022, 0.5, 0.887298334620741688517926539980)
const QWEIGHTS = (0.277777777777777777777777777779, 0.444444444444444444444444444444, 0.277777777777777777777777777779)

@inline state_idx(ll, k, i, nx, nz, hs) = Int(ll) * Int(nz + 2hs) * Int(nx + 2hs) + Int(k) * Int(nx + 2hs) + Int(i) + 1
@inline flux_idx(ll, k, i, nx, nz) = Int(ll) * Int(nz + 1) * Int(nx + 1) + Int(k) * Int(nx + 1) + Int(i) + 1
@inline tend_idx(ll, k, i, nx, nz) = Int(ll) * Int(nz) * Int(nx) + Int(k) * Int(nx) + Int(i) + 1

