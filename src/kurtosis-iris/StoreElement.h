#pragma once

#include <cstdint>
#include <sstream>
#include <string>

#include "StoreTypedefs.h"

struct storeElement {
  int32_t tag;
  metric_type metric;
  ullint time;
  store_value_type value;

  storeElement() : tag(0), metric(0), time(0), value(0) {}
  storeElement(int _tag, metric_type _metric, ullint _time, store_value_type _value)
      : tag(_tag), metric(_metric), time(_time), value(_value) {}

  std::string toString() const
  {
    std::ostringstream stream;
    stream << "[" << tag << ", " << metric << ", " << time << ", " << value << "]";
    return stream.str();
  }
};
