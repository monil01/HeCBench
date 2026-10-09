#ifndef BWT_IRIS_BWT_HPP
#define BWT_IRIS_BWT_HPP

#include <string>
#include <utility>

const char ETX = '\0';

std::pair<std::string, int*> bwt_with_suffix_array(const std::string sequence);
std::string bwt(const std::string sequence);

#endif
