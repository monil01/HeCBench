#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define NUMTHREADS 256

struct StructureAtom {
  double x;
  double y;
  double z;
  double epsilon;
  double sigma;
};

static bool load_structure(const char* path, std::vector<StructureAtom>& atoms, double& L)
{
  std::ifstream materialfile(path);
  if (materialfile.fail()) {
    std::printf("Failed to import file %s.\n", path);
    return false;
  }

  std::map<std::string, double> epsilons;
  epsilons["Zn"] = 96.152688;
  epsilons["O"] = 66.884614;
  epsilons["C"] = 88.480032;
  epsilons["H"] = 57.276566;

  std::map<std::string, double> sigmas;
  sigmas["Zn"] = 3.095775;
  sigmas["O"] = 3.424075;
  sigmas["C"] = 3.580425;
  sigmas["H"] = 3.150565;

  std::string line;
  std::getline(materialfile, line);
  std::istringstream istream(line);
  istream >> L;
  std::printf("L = %f\n", L);

  std::getline(materialfile, line);
  std::getline(materialfile, line);
  int natoms = 0;
  istream.str(line);
  istream.clear();
  istream >> natoms;
  std::printf("%d atoms\n", natoms);

  std::getline(materialfile, line);
  atoms.resize(natoms);
  for (int i = 0; i < natoms; i++) {
    std::getline(materialfile, line);
    istream.str(line);
    istream.clear();

    int atomno = 0;
    double xf = 0.0;
    double yf = 0.0;
    double zf = 0.0;
    std::string element;
    istream >> atomno >> element >> xf >> yf >> zf;

    atoms[i].x = L * xf;
    atoms[i].y = L * yf;
    atoms[i].z = L * zf;
    atoms[i].epsilon = epsilons[element];
    atoms[i].sigma = sigmas[element];
  }
  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: ./%s <material file> <ninsertions>\n", argv[0]);
    return EXIT_FAILURE;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::vector<StructureAtom> structure_atoms;
  double L = 0.0;
  if (!load_structure(argv[1], structure_atoms, L)) {
    platform.finalize();
    return EXIT_FAILURE;
  }

  int ncycles = std::atoi(argv[2]);
  int natoms = static_cast<int>(structure_atoms.size());
  int nblocks = 1024;
  int insertions_per_cycle = nblocks * NUMTHREADS;
  int ninsertions = ncycles * insertions_per_cycle;

  std::vector<double> boltzmann_factors(insertions_per_cycle);
  iris::DMem mem_atoms(structure_atoms.data(), structure_atoms.size() * sizeof(StructureAtom));
  iris::DMem mem_factors(boltzmann_factors.data(), boltzmann_factors.size() * sizeof(double));

  const size_t gws[1] = { static_cast<size_t>(insertions_per_cycle) };
  const size_t lws[1] = { NUMTHREADS };

  double total_time = 0.0;
  double KH = 0.0;
  for (int cycle = 0; cycle < ncycles; cycle++) {
    auto start = std::chrono::steady_clock::now();
    iris::Task task;
    void* params[4] = { &mem_factors, &mem_atoms, &natoms, &L };
    int pinfo[4] = { iris_w, iris_r, sizeof(natoms), sizeof(L) };
    task.kernel("insertions", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4, params, pinfo);
    task.flush_out(mem_factors);
    task.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    mem_factors.fetch(boltzmann_factors.data());
    for (int i = 0; i < insertions_per_cycle; i++) {
      KH += boltzmann_factors[i];
    }
  }

  const double T = 298.0;
  const double R = 8.314;
  KH = KH / ninsertions;
  KH = KH / (R * T);
  std::printf("Used %d blocks with %d thread each\n", nblocks, NUMTHREADS);
  std::printf("Henry constant = %e mol/(m3 - Pa)\n", KH);
  std::printf("Number of actual insertions: %d\n", ninsertions);
  std::printf("Number of times we called the device kernel: %d\n", ncycles);
  std::printf("Average kernel execution time %f (s)\n", (total_time * 1e-9) / ncycles);
  std::printf("PASS\n");

  platform.finalize();
  return EXIT_SUCCESS;
}
