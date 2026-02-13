#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace core {

template<int N>
struct Vec {
  std::array<double, N> a{};

  double& operator[](int i) { return a[i]; }
  const double& operator[](int i) const { return a[i]; }
};

template<int R, int C>
struct Mat {
  // row-major
  std::array<double, R*C> a{};

  double& operator()(int r, int c) { return a[r*C + c]; }
  const double& operator()(int r, int c) const { return a[r*C + c]; }

  static Mat<R,C> zero() { return Mat<R,C>{}; }
};

using Vec3 = Vec<3>;
using Vec4 = Vec<4>;
using Vec5 = Vec<5>;
using Mat4 = Mat<4,4>;
using Mat5 = Mat<5,5>;

template<int N>
inline double dot(const Vec<N>& x, const Vec<N>& y) {
  double s = 0.0;
  for (int i=0;i<N;i++) s += x[i]*y[i];
  return s;
}

template<int N>
inline double norm(const Vec<N>& x) {
  return std::sqrt(std::max(0.0, dot(x,x)));
}

template<int N>
inline Vec<N> add(const Vec<N>& x, const Vec<N>& y) {
  Vec<N> r;
  for (int i=0;i<N;i++) r[i] = x[i] + y[i];
  return r;
}

template<int N>
inline Vec<N> sub(const Vec<N>& x, const Vec<N>& y) {
  Vec<N> r;
  for (int i=0;i<N;i++) r[i] = x[i] - y[i];
  return r;
}

template<int N>
inline Vec<N> mul(double s, const Vec<N>& x) {
  Vec<N> r;
  for (int i=0;i<N;i++) r[i] = s*x[i];
  return r;
}

template<int R, int C>
inline Vec<R> mul(const Mat<R,C>& M, const Vec<C>& x) {
  Vec<R> y;
  for (int r=0;r<R;r++) {
    double s = 0.0;
    for (int c=0;c<C;c++) s += M(r,c)*x[c];
    y[r] = s;
  }
  return y;
}

template<int R, int C>
inline Vec<R> col(const Mat<R,C>& M, int c) {
  Vec<R> v;
  for (int r=0;r<R;r++) v[r] = M(r,c);
  return v;
}

template<int R, int C>
inline void set_col(Mat<R,C>& M, int c, const Vec<R>& v) {
  for (int r=0;r<R;r++) M(r,c) = v[r];
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
  Vec3 c;
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
  return c;
}

} // namespace core
