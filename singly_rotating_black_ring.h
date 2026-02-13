#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include "vec_mat.h"
#include "rk4_adaptive.h"

namespace singly_rotating_black_ring {

using core::Vec3; using core::Vec4; using core::Vec5;
using core::Mat4; using core::Mat5;

// Convenience: conical-defect-free balance condition for the neutral singly-spinning ring.
inline constexpr double balanced_lambda(double nu) {
  return (2.0 * nu) / (1.0 + nu * nu);
}

struct Params {
  double R = 10.0;
  double nu = 0.6;
  double lambda = 0.0; // set explicitly (e.g. balanced_lambda(nu))
};

struct RenderOpts {
  // stop when 1 + nu*y <= horizon_eps (with y = -cosh(eta))
  double horizon_eps = 1e-5;
  // stop when rho_cart >= r_escape
  double r_escape = 500.0;
};

struct RayConsts {
  double E = 0.0;
  double Phi = 0.0;
  double Psi = 0.0;
};

// ------------------------------
// Coordinate chart (exterior):
//   x = tanh(chi),   chi ∈ (-∞, +∞)   -> x ∈ (-1, 1)
//   y = -cosh(eta),  eta ∈ [0, +∞)    -> y ≤ -1
//
// Infinity corner (x,y)->(-1,-1) corresponds to (chi,eta)->(-∞,0).
// ------------------------------
inline double x_from_chi(double chi) { return std::tanh(chi); }
inline double y_from_eta(double eta) { return -std::cosh(eta); }

inline double chi_from_x(double x) {
  // x must lie in (-1,1); clamp away from endpoints for numeric safety
  constexpr double eps = 1e-15;
  x = std::clamp(x, -1.0 + eps, 1.0 - eps);
  return std::atanh(x);
}

inline double eta_from_y(double y) {
  // exterior has y <= -1; enforce -y >= 1 for acosh
  const double a = std::max(1.0, -y);
  return std::acosh(a);
}

// State for Hamiltonian null geodesics in (t,chi,eta,phi,psi):
//   y = [t, chi, eta, phi, psi, p_chi, p_eta]
using Y7 = std::array<double,7>;

// ------------------------------------------------------------
// Chart folding / projector for the (chi,eta) exterior chart.
//
// Only eta has a double-cover: y=-cosh(eta) is even.
// If eta flips sign, preserve the same embedded point by shifting
//   psi -> psi + pi
// and flip conjugate momentum p_eta for Hamiltonian consistency.
// ------------------------------------------------------------
inline void fold_chieta(Y7& y)
{
  double& eta  = y[2];
  double& psi  = y[4];
  double& peta = y[6];

  if (std::isfinite(eta) && eta < 0.0) {
    eta  = -eta;
    peta = -peta;
    psi += core::RK_PI;
  }
}

struct Frame5 {
  Mat5 E5;   // columns: [u, e1, e2, e3, e4] in (t,chi,eta,phi,psi)
  Mat5 g;    // metric at camera point in (t,chi,eta,phi,psi)
  Vec5 q;    // [t,chi,eta,phi,psi] with t=0 at the camera
  Vec4 p4;   // camera position in R^4
};

// Build full 5D orthonormal frame (your conventions):
//   psi is azimuth in (x1,x2)
//   phi is azimuth in (x3,x4)
Frame5 build_camera_frame(const Params& P, const Vec3& pos, const Vec3& up, const Vec4& v);

// Initialize null ray from local frame:
// n_cam = [right, up, forward, w] direction in camera frame (need not be unit).
// Outputs y0 and conserved quantities (E,Phi,Psi).
bool init_null_ray_from_frame(const Frame5& F, const Vec4& n_cam, Y7& y0, RayConsts& cst);

// RHS for null geodesics (Hamiltonian first-order form) in (chi,eta) chart.
void rhs(const Params& P, const RayConsts& C, double s, const Y7& y, Y7& dyds);

// Event checker: returns 0 none, 1 horizon, 2 escape, 3 invalid
int check_event(const Params& P, const RenderOpts& R, double s, const Y7& y);

// Trace one ray with adaptive RK4 + events
core::RKResult<7> trace_ray(const Params& P, const RenderOpts& R, const core::RKOpts& O,
                           const Y7& y0, const RayConsts& C, double sMax);

} // namespace singly_rotating_black_ring





