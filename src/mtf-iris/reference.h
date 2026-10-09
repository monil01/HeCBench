#include <algorithm>
#include <vector>

std::vector<char> reference(const std::vector<char>& word)
{
  std::vector<char> list(256);
  std::vector<char> d_list(256);
  std::vector<char> d_word(word);

  size_t counter;

  for (counter = 0; counter < word.size(); counter++) {
    std::copy(list.begin(), list.end(), d_list.begin());
    auto w = d_word[counter];
    auto iter = find(d_list.begin(), d_list.end(), d_word[counter]);
    if (d_list[0] != w) {
      const size_t count = static_cast<size_t>(std::distance(d_list.begin(), iter));
      const size_t shifted = std::min<size_t>(count, list.size() - 1);
      std::copy(d_list.begin(), d_list.begin() + shifted, list.begin() + 1);
      list[0] = w;
    }
  }

  std::copy(list.begin(), list.end(), d_list.begin());
  for (counter = 0; counter < list.size(); counter++) {
    auto iter = std::find(d_word.begin(), d_word.end(), d_list[counter]);
    while (iter != d_word.end()) {
      *iter = counter;
      iter = std::find(iter + 1, d_word.end(), d_list[counter]);
    }
  }
  return d_word;
}
