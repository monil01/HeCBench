#pragma once
#include <cmath>
struct PayoffCall
{
  double m_K;
  inline PayoffCall(double K) : m_K(K) {}
  inline double operator()(double S) const { return fmax(S - m_K, 0.0); }
  inline int is_in_the_money(double S) const { return S > m_K; }
};

struct PayoffPut
{
  double m_K;
  inline PayoffPut(double K) : m_K(K) {}
  inline double operator()(double S) const { return fmax(m_K - S, 0.0); }
  inline int is_in_the_money(double S) const { return S < m_K; }
};
