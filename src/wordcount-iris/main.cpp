#include <iris/iris.hpp>

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

static bool is_alpha(const char c) {
  return c >= 'A' && c <= 'z';
}

struct is_word_start {
  bool operator()(const char& left, const char& right) const {
    return is_alpha(right) && !is_alpha(left);
  }
};

static int word_count_reference(const std::vector<char>& input) {
  if (input.empty()) return 0;

  int wc = std::inner_product(input.cbegin(), input.cend() - 1,
                              input.cbegin() + 1, 0, std::plus<int>(),
                              is_word_start());
  if (is_alpha(input.front())) wc++;
  return wc;
}

static int word_count_device(const std::vector<char>& input) {
  if (input.empty()) return 0;

  std::vector<unsigned char> starts(input.size(), 0);
  iris::DMem mem_input(const_cast<char*>(input.data()),
                       input.size() * sizeof(char));
  iris::DMem mem_starts(starts.data(), starts.size() * sizeof(unsigned char));

  const size_t n = input.size();
  const size_t block = 256;
  const size_t grid = ((n + block - 1) / block) * block;

  iris::Task task;
  void* params[] = {&mem_starts, &mem_input, const_cast<size_t*>(&n)};
  int pinfo[] = {iris_w, iris_r, sizeof(n)};
  size_t gws[] = {grid};
  size_t lws[] = {block};
  task.kernel("wordcount_mark_starts", 1, nullptr, gws, lws, 3, params, pinfo);
  task.flush_out(mem_starts);
  task.submit(iris_gpu, nullptr, 1);

  return std::accumulate(starts.begin(), starts.end(), 0);
}

int main(int argc, char* argv[]) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <repeat>\n";
    platform.finalize();
    return 1;
  }
  const int repeat = std::atoi(argv[1]);
  if (repeat <= 0) {
    std::cerr << "repeat must be positive\n";
    platform.finalize();
    return 1;
  }

  const char raw_input[] =
      "  But the raven, sitting lonely on the placid bust, spoke only,\n"
      "  That one word, as if his soul in that one word he did outpour.\n"
      "  Nothing further then he uttered - not a feather then he fluttered -\n"
      "  Till I scarcely more than muttered `Other friends have flown before -\n"
      "  On the morrow he will leave me, as my hopes have flown before.'\n"
      "  Then the bird said, `Nevermore.'\n";

  std::cout << "Text sample:" << std::endl;
  std::cout << raw_input << std::endl;

  std::vector<char> input(raw_input, raw_input + sizeof(raw_input));

  int wc = word_count_reference(input);
  std::cout << "Host: Text sample contains " << wc << " words" << std::endl;

  wc = word_count_device(input);
  std::cout << "Device: Text sample contains " << wc << " words" << std::endl;

  std::cout << "Test word count with random inputs\n";
  srand(123);
  bool ok = true;
  const char tab[] = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const int size = std::strlen(tab);

  for (size_t i = 1; i <= 100000000; i = i * 10) {
    std::vector<char> random_input(i);
    for (size_t c = 0; c < i; c++) random_input[c] = tab[rand() % size];
    if (word_count_reference(random_input) != word_count_device(random_input)) {
      ok = false;
      break;
    }
  }
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  const size_t len = 1024 * 1024 * 256;
  std::vector<char> random_input(len);
  for (size_t c = 0; c < len; c++) random_input[c] = tab[rand() % size];

  std::cout << "Performance evaluation for random texts of character length "
            << len << std::endl;
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) word_count_device(random_input);

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count();
  std::cout << "Average time of word count: " << time * 1e-9f / repeat
            << " (s)\n";

  platform.finalize();
  return ok ? 0 : 1;
}
