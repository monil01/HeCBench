#include <iris/iris.hpp>
#include "hip_backend.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "multians.h"

#define NUM_SYMBOLS 256
#define NUM_STATES 1024
#define SEED 5
#define SUBSEQUENCE_SIZE 4
#define THREADS_PER_BLOCK 128

static void submit_kernel(const char* name,
                          size_t grid,
                          size_t block,
                          int nparams,
                          void** params,
                          int* pinfo,
                          std::initializer_list<iris::DMem*> flushes = {}) {
  iris::Task task;
  size_t gws[] = {grid * block};
  size_t lws[] = {block};
  iris_check(task.kernel(name, 1, nullptr, gws, lws, nparams, params, pinfo), name);
  for (iris::DMem* mem : flushes) iris_check(task.flush_out(*mem), "flush");
  iris_check(task.submit(hip_device, nullptr, 1), name);
}

static void iris_decode(iris::DMem& input_mem,
                        size_t input_size,
                        iris::DMem& output_mem,
                        size_t output_size,
                        iris::DMem& table_mem,
                        iris::DMem& sync_info_mem,
                        iris::DMem& output_sizes_mem,
                        iris::DMem& sequence_synced_mem,
                        std::vector<std::uint32_t>& output_sizes,
                        std::vector<std::uint8_t>& sequence_synced,
                        STATE_TYPE initial_state,
                        std::uint32_t initial_bit,
                        std::uint32_t number_of_states,
                        size_t max_codeword_length,
                        size_t preferred_subsequence_size,
                        size_t threads_per_block) {
  std::uint32_t num_subseq = static_cast<std::uint32_t>(SDIV(input_size, preferred_subsequence_size));
  std::uint32_t num_sequences = static_cast<std::uint32_t>(SDIV(num_subseq, threads_per_block));
  std::uint32_t bits_in_unit = sizeof(UNIT_TYPE) * 8;
  std::uint32_t table_size = static_cast<std::uint32_t>(max_codeword_length);
  std::uint32_t subsequence_size = static_cast<std::uint32_t>(preferred_subsequence_size);
  std::uint32_t block_size = static_cast<std::uint32_t>(threads_per_block);

  {
    void* params[] = {&subsequence_size, &num_subseq, &table_size, &input_mem, &table_mem, &sync_info_mem,
                      &bits_in_unit, &number_of_states, &initial_state, &initial_bit};
    int pinfo[] = {sizeof(subsequence_size), sizeof(num_subseq), sizeof(table_size), iris_r, iris_r, iris_rw,
                   sizeof(bits_in_unit), sizeof(number_of_states), sizeof(initial_state), sizeof(initial_bit)};
    submit_kernel("phase1_decode_subseq", num_sequences, block_size, 10, params, pinfo);
  }

  bool blocks_synchronised = true;
  do {
    void* params[] = {&subsequence_size, &num_subseq, &table_size, &num_sequences, &input_mem, &table_mem,
                      &sync_info_mem, &sequence_synced_mem, &bits_in_unit, &number_of_states, &initial_state};
    int pinfo[] = {sizeof(subsequence_size), sizeof(num_subseq), sizeof(table_size), sizeof(num_sequences),
                   iris_r, iris_r, iris_rw, iris_w, sizeof(bits_in_unit), sizeof(number_of_states),
                   sizeof(initial_state)};
    submit_kernel("phase2_synchronise_blocks", num_sequences, block_size, 11, params, pinfo,
                  {&sequence_synced_mem, &sync_info_mem});

    bool zero_found = false;
    for (size_t i = 1; i + 1 < num_sequences; ++i) {
      if (sequence_synced[i] == 0) {
        zero_found = true;
        break;
      }
    }
    blocks_synchronised = !zero_found;
  } while (!blocks_synchronised);

  {
    void* params[] = {&num_subseq, &sync_info_mem, &output_sizes_mem};
    int pinfo[] = {sizeof(num_subseq), iris_r, iris_w};
    submit_kernel("phase3_copy_num_symbols_from_sync_points_to_aux", num_sequences, block_size, 3, params, pinfo,
                  {&output_sizes_mem});
  }

  // Keep flushed counts and uploaded offsets in distinct host allocations.
  // Two live DMem objects must not register the same host address.
  std::vector<std::uint32_t> offsets(output_sizes.size(), 0);
  std::uint32_t running = 0;
  for (size_t i = 0; i < num_subseq; ++i) {
    std::uint32_t current = output_sizes[i];
    offsets[i] = running;
    running += current;
  }

  {
    iris::DMem output_sizes_scan_mem(offsets.data(), offsets.size() * sizeof(std::uint32_t));
    void* params[] = {&num_subseq, &sync_info_mem, &output_sizes_scan_mem};
    int pinfo[] = {sizeof(num_subseq), iris_rw, iris_r};
    submit_kernel("phase3_copy_num_symbols_from_aux_to_sync_points", num_sequences, block_size, 3, params, pinfo,
                  {&sync_info_mem});
  }

  std::uint32_t output_size_u32 = static_cast<std::uint32_t>(output_size);
  {
    void* params[] = {&subsequence_size, &num_subseq, &table_size, &input_mem, &output_mem, &output_size_u32,
                      &table_mem, &sync_info_mem, &bits_in_unit, &number_of_states, &initial_state, &initial_bit};
    int pinfo[] = {sizeof(subsequence_size), sizeof(num_subseq), sizeof(table_size), iris_r, iris_rw,
                   sizeof(output_size_u32), iris_r, iris_r, sizeof(bits_in_unit), sizeof(number_of_states),
                   sizeof(initial_state), sizeof(initial_bit)};
    submit_kernel("phase4_decode_write_output", num_sequences, block_size, 12, params, pinfo, {&output_mem});
  }
}

static bool run(long int input_size) {
  bool ok = true;
  std::cout << "\u03bb | compressed size (bytes) | ";
  std::cout << std::endl << std::endl;

  auto start = std::chrono::steady_clock::now();

  for (float lambda = 0.1f; lambda < 2.5f; lambda += 0.16) {
    std::cout << std::left << std::setw(5) << lambda << std::setfill(' ');

    auto dist = ANSTableGenerator::generate_distribution(
        SEED, NUM_SYMBOLS, NUM_STATES, [&](double x) { return lambda * exp(-lambda * x); });
    auto random_data = ANSTableGenerator::generate_test_data(dist.dist, input_size, NUM_STATES, SEED);
    auto table = ANSTableGenerator::generate_table(dist.prob, dist.dist, nullptr, NUM_SYMBOLS, NUM_STATES);
    auto encoder_table = ANSTableGenerator::generate_encoder_table(table);
    auto decoder_table = ANSTableGenerator::get_decoder_table(encoder_table);
    auto input_buffer = ANSEncoder::encode(random_data->data(), input_size, encoder_table);
    auto output_buffer = std::make_shared<CUHDOutputBuffer>(input_size);

    size_t compressed_size = input_buffer->get_compressed_size();
    std::vector<UNIT_TYPE> input_data(compressed_size + 4);
    std::memcpy(input_data.data(), input_buffer->get_compressed_data(), sizeof(UNIT_TYPE) * (compressed_size + 4));

    size_t decoder_words = decoder_table->get_size() * sizeof(CUHDCodetableItem) / sizeof(std::uint32_t);
    std::vector<std::uint32_t> decoder_data(decoder_words);
    std::memcpy(decoder_data.data(), reinterpret_cast<std::uint32_t*>(decoder_table->get()),
                decoder_words * sizeof(std::uint32_t));

    size_t output_buffer_bytes = sizeof(SYMBOL_TYPE) * output_buffer->get_uncompressed_size();
    std::vector<SYMBOL_TYPE> output_data(output_buffer->get_uncompressed_size());
    if (std::getenv("HECBENCH_VALIDATE_SENTINEL"))
      std::fill(output_data.begin(), output_data.end(), static_cast<SYMBOL_TYPE>(165));
    size_t num_subseq = SDIV(compressed_size, SUBSEQUENCE_SIZE);
    size_t num_blocks = SDIV(num_subseq, THREADS_PER_BLOCK);
    std::vector<uint4> sync_info(num_subseq + 4);
    std::vector<std::uint32_t> output_sizes(num_subseq + 4);
    std::vector<std::uint8_t> sequence_synced(num_blocks);

    iris::DMem input_mem(input_data.data(), input_data.size() * sizeof(UNIT_TYPE));
    iris::DMem table_mem(decoder_data.data(), decoder_data.size() * sizeof(std::uint32_t));
    iris::DMem output_mem(output_data.data(), output_buffer_bytes);
    iris::DMem sync_info_mem(sync_info.data(), sync_info.size() * sizeof(uint4));
    iris::DMem output_sizes_mem(output_sizes.data(), output_sizes.size() * sizeof(std::uint32_t));
    iris::DMem sequence_synced_mem(sequence_synced.data(), sequence_synced.size() * sizeof(std::uint8_t));

    iris_decode(input_mem, input_buffer->get_compressed_size(), output_mem, output_buffer->get_uncompressed_size(),
                table_mem, sync_info_mem, output_sizes_mem, sequence_synced_mem, output_sizes, sequence_synced,
                input_buffer->get_first_state(), input_buffer->get_first_bit(), decoder_table->get_num_entries(),
                11, SUBSEQUENCE_SIZE, THREADS_PER_BLOCK);

    std::memcpy(output_buffer->get_decompressed_data().get(), output_data.data(), output_buffer_bytes);
    output_buffer->reverse();
    if (const char* path = std::getenv("HECBENCH_DUMP")) {
      FILE* f = fopen(path, "ab");
      if (!f || fwrite(output_buffer->get_decompressed_data().get(), 1, input_size, f) != (size_t)input_size || fclose(f)) exit(3);
    }
    if (std::getenv("HECBENCH_VALIDATE_SENTINEL"))
      std::cout << " expected_first=" << int((*random_data)[0])
                << " decoded_first=" << int(output_buffer->get_decompressed_data()[0]) << " ";

    if (cuhd::CUHDUtil::equals(random_data->data(), output_buffer->get_decompressed_data().get(), input_size)) {
      std::cout << std::left << std::setw(10) << input_buffer->get_compressed_size() * sizeof(UNIT_TYPE)
                << std::setfill(' ') << std::endl;
    } else {
      ok = false;
      std::cout << "********* MISMATCH ************" << std::endl;
      std::cout << std::left << std::setw(10) << input_buffer->get_compressed_size() * sizeof(UNIT_TYPE)
                << std::setfill(' ') << std::endl;
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Total elapsed time " << time * 1e-9f << " (s)\n";
  return ok;
}

int main(int argc, char** argv) {
  const char* bin = argv[0];
  auto print_help = [&]() { std::cout << "USAGE: " << bin << "<size of input in megabytes> " << std::endl; };
  if (argc < 2) {
    print_help();
    return 1;
  }

  const long int size = atoi(argv[1]) * 1024 * 1024;
  if (size < 1) {
    print_help();
    return 1;
  }
  assert(SUBSEQUENCE_SIZE % 4 == 0);

  iris::Platform platform;
  iris_check(platform.init(&argc, &argv, 1), "initialize");
  select_hip_device();
  bool ok = run(size);
  platform.finalize();
  return ok ? 0 : 1;
}
