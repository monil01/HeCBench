#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>
struct Option { std::string inputFile, outputFile; float threshold; int wordLength; };
struct Read { std::string data, name; };
void checkOption(int argc,char** argv,Option& option);
bool readFile(std::vector<Read>& reads,Option& option);
