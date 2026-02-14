#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include "vec_mat.h"
#include "rk4_adaptive.h"

namespace singly_rotating_black_ring {

using core::Vec3; using core::Vec4; using core::Vec5;
using core::Mat4; using core::Mat5;

inline constexpr double balanced_lambda(double nu) {
  return (2.0 * nu) / (1.0 + nu * nu);
}

struct Params {
  double R = 10.0;
  double nu = 0.6;
  double lambda = 0.0;
};

struct RenderOpts {
  double horizon_eps = 1e-5;
  double r_escape = 500.0;

  // 0,1,2
  int rescale_pow = 0;
};

struct RayConsts {
  double E = 0.0;
  double Phi = 0.0;
  double Psi = 0.0;
};

// State: [t, x, y, phi, psi, p_x, p_y]
using Y7 = std::array<double,7>;

inline void project_xy(Y7& y)
{
  y[3] = core::wrap_pi(y[3]);
  y[4] = core::wrap_pi(y[4]);
}

struct Frame5 {
  Mat5 E5;   // columns: [u, e1, e2, e3, e4] in (t,x,y,phi,psi)
  Mat5 g;    // metric at camera point in (t,x,y,phi,psi)
  Vec5 q;    // [t,x,y,phi,psi] with t=0 at the camera
  Vec4 p4;   // camera position in R^4
};

Frame5 build_camera_frame(const Params& P, const Vec3& pos, const Vec3& up, const Vec4& v);
bool init_null_ray_from_frame(const Frame5& F, const Vec4& n_cam, Y7& y0, RayConsts& cst);

void rhs(const Params& P, const RayConsts& C, double s, const Y7& y, Y7& dyds);
int check_event(const Params& P, const RenderOpts& R, double s, const Y7& y);

core::RKResult<7> trace_ray(const Params& P, const RenderOpts& R, const core::RKOpts& O,
                           const Y7& y0, const RayConsts& C, double sMax);

// NEW: dump full adaptive trajectory (accept/reject/bad) to CSV.
core::RKResult<7> trace_ray_dump_csv(const Params& P, const RenderOpts& R, const core::RKOpts& O,
                                     const Y7& y0, const RayConsts& C, double sMax,
                                     const char* csv_path);

} // namespace singly_rotating_black_ring







