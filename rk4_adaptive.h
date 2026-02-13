#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>

namespace core {

struct RKOpts {
  double rtol = 1e-6;
  double atol = 1e-8;
  double h0   = 0.5;
  double hmin = 1e-12;
  double hmax = 5.0;
  int    max_steps = 200000;

  double safety = 0.9;
  double fac_min = 0.2;
  double fac_max = 5.0;

  // If true, allow accepting at hmin even if errInf > 1 (ONLY if errInf is finite).
  // Default false: return RKReason::Hmin instead of marching into potentially invalid states.
  bool force_accept_hmin = false;
};

enum class RKReason { Done, Event, Hmin, MaxSteps, NaN };

template<size_t N>
struct RKResult {
  double s_end = 0.0;
  std::array<double,N> y_end{};
  int event = 0;            // 0 none, otherwise event id
  int n_accept = 0;
  int n_reject = 0;
  RKReason reason = RKReason::Done;
};

// ---------- helpers ----------
static constexpr double RK_PI = 3.14159265358979323846;
static constexpr double RK_TWOPI = 2.0 * RK_PI;

// For RK4 step-doubling: (y_2half - y_full)/15 estimates the local error of y_2half.
static constexpr double RK_ERR_SCALE = 1.0 / 15.0;

// Wrap to [-pi, pi]
inline double wrap_pi(double a) {
  return std::remainder(a, RK_TWOPI);
}

// No-op state projector (default).
struct IdentityProjector {
  template<class Arr>
  void operator()(Arr&) const noexcept {}
};

// ---------- fixed-step RK4 (PROJECTED) ----------
template<size_t N, class RHS, class Projector>
inline void rk4_step(RHS&& rhs,
                     Projector&& proj,
                     double s,
                     const std::array<double,N>& y,
                     double h,
                     std::array<double,N>& yout)
{
  std::array<double,N> k1{}, k2{}, k3{}, k4{}, yt{}, ybase = y;

  // Ensure base state is on the intended chart/manifold.
  proj(ybase);

  rhs(s, ybase, k1);

  for (size_t i=0;i<N;i++) yt[i] = ybase[i] + 0.5*h*k1[i];
  proj(yt);
  rhs(s + 0.5*h, yt, k2);

  for (size_t i=0;i<N;i++) yt[i] = ybase[i] + 0.5*h*k2[i];
  proj(yt);
  rhs(s + 0.5*h, yt, k3);

  for (size_t i=0;i<N;i++) yt[i] = ybase[i] + h*k3[i];
  proj(yt);
  rhs(s + h, yt, k4);

  for (size_t i=0;i<N;i++) {
    yout[i] = ybase[i] + (h/6.0)*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
  }

  // Keep output on-chart too.
  proj(yout);
}

// ---------- fixed-step RK4 (legacy, unprojected interface) ----------
template<size_t N, class RHS>
inline void rk4_step(RHS&& rhs,
                     double s,
                     const std::array<double,N>& y,
                     double h,
                     std::array<double,N>& yout)
{
  IdentityProjector proj{};
  rk4_step<N>(rhs, proj, s, y, h, yout);
}

// ---------- default error estimator ----------
template<size_t N>
inline double default_err_inf_raw(const std::array<double,N>& y,
                                  const std::array<double,N>& y_full,
                                  const std::array<double,N>& y_2half,
                                  const RKOpts& opt)
{
  double errInf = 0.0;

  for (size_t i=0;i<N;i++) {
    if constexpr (N == 7) {
      // For your black-ring state: ignore t, phi, psi in the *default* norm.
      if (i == 0 || i == 3 || i == 4) continue;
    }

    double err = y_2half[i] - y_full[i];
    double sc  = opt.atol + opt.rtol * std::max(std::abs(y[i]), std::abs(y_2half[i]));
    errInf = std::max(errInf, std::abs(err)/sc);
  }

  return errInf;
}

// ---------- adaptive RK4 with custom error estimator + PROJECTOR ----------
// Signature (note proj is LAST to avoid overload collisions):
//   rk4_adaptive(rhs, ev, err_est, s0, y0, sMax, opt, proj)
template<size_t N, class RHS, class EventCheck, class ErrorEstimator, class Projector>
inline RKResult<N> rk4_adaptive(RHS&& rhs,
                               EventCheck&& ev,
                               ErrorEstimator&& err_est,
                               double s0,
                               const std::array<double,N>& y0,
                               double sMax,
                               const RKOpts& opt,
                               Projector&& proj)
{
  RKResult<N> out;
  out.s_end = s0;

  double s = s0;
  std::array<double,N> y = y0;
  proj(y);

  out.y_end = y;

  // Initial event check (on projected state)
  {
    int ev0 = ev(s, y);
    if (ev0 != 0) {
      out.reason = RKReason::Event;
      out.event = ev0;
      out.s_end = s;
      out.y_end = y;
      return out;
    }
  }

  double span = sMax - s0;
  double h = std::clamp(opt.h0 > 0 ? opt.h0 : span/1000.0, opt.hmin, opt.hmax);

  int step = 0;
  for (step=0; step<opt.max_steps; ++step) {
    if (s >= sMax) { out.reason = RKReason::Done; break; }

    // Ensure state is on-chart before trying a step.
    proj(y);

    // Don't take a step past sMax
    h = std::min(h, sMax - s);

    // If what's left is smaller than hmin, treat as Done (raytracing-friendly).
    if (h < opt.hmin) { out.reason = RKReason::Done; break; }

    std::array<double,N> y_full{}, y_half{}, y_2half{};
    rk4_step<N>(rhs, proj, s, y, h, y_full);

    rk4_step<N>(rhs, proj, s, y, 0.5*h, y_half);
    rk4_step<N>(rhs, proj, s+0.5*h, y_half, 0.5*h, y_2half);

    // If a trial step produces NaN/Inf, treat it as a REJECT (step too big),
    // shrink h, and retry. Only fail if we cannot shrink further.
    bool bad = false;
    for (size_t i=0;i<N;i++) {
      if (!std::isfinite(y_full[i]) || !std::isfinite(y_2half[i])) { bad = true; break; }
    }
    if (bad) {
      out.n_reject++;

      if (h <= opt.hmin * 1.0000000001) {
        out.reason = RKReason::NaN;
        out.s_end = s;
        out.y_end = y;
        return out;
      }

      // Shrink aggressively on bad steps.
      h = std::max(h * opt.fac_min, opt.hmin);
      continue;
    }

    // Convert raw difference norm into an estimate of local error (divide by 15).
    double errInf = err_est(y, y_full, y_2half, opt) * RK_ERR_SCALE;
    if (!std::isfinite(errInf)) errInf = std::numeric_limits<double>::infinity();

    if (errInf <= 1.0) {
      // ACCEPT: use the better approximation y_2half directly.
      std::array<double,N> y_new = y_2half;

      // Re-project (caller controls any "hygiene" like angle wrapping).
      proj(y_new);

      double s_new = s + h;

      int ev_id = ev(s_new, y_new);
      if (ev_id != 0) {
        out.reason = RKReason::Event;
        out.event = ev_id;
        out.s_end = s_new;
        out.y_end = y_new;
        out.n_accept++;
        return out;
      }

      // Commit
      s = s_new;
      y = y_new;
      out.n_accept++;

      // Step update (order 4 => local error ~ h^5 => exponent 1/5)
      double fac;
      if (errInf == 0.0) fac = opt.fac_max;
      else {
        fac = opt.safety * std::pow(errInf, -0.2);
        fac = std::clamp(fac, opt.fac_min, opt.fac_max);
      }
      h = std::clamp(h*fac, opt.hmin, opt.hmax);

      out.s_end = s;
      out.y_end = y;
    } else {
      // REJECT
      out.n_reject++;

      // At (or below) hmin: either stop, or optionally force-accept if finite.
      if (h <= opt.hmin * 1.0000000001) {
        if (opt.force_accept_hmin && std::isfinite(errInf)) {
          std::array<double,N> y_new = y_2half;

          proj(y_new);

          double s_new = s + h;

          int ev_id = ev(s_new, y_new);
          if (ev_id != 0) {
            out.reason = RKReason::Event;
            out.event = ev_id;
            out.s_end = s_new;
            out.y_end = y_new;
            out.n_accept++;
            return out;
          }

          s = s_new;
          y = y_new;
          out.n_accept++;

          h = opt.hmin; // can't go smaller; keep limping if caller asked for it
          out.s_end = s;
          out.y_end = y;
          continue;
        } else {
          out.reason = RKReason::Hmin;
          break;
        }
      }

      // Shrink step
      double fac = opt.safety * std::pow(errInf, -0.2);
      fac = std::clamp(fac, opt.fac_min, 0.9);
      h = std::max(h*fac, opt.hmin);
    }
  }

  out.s_end = s;
  out.y_end = y;

  if (step >= opt.max_steps && out.reason == RKReason::Done) {
    out.reason = RKReason::MaxSteps;
  }

  return out;
}

// ---------- adaptive RK4 with custom error estimator (legacy, no projector) ----------
template<size_t N, class RHS, class EventCheck, class ErrorEstimator>
inline RKResult<N> rk4_adaptive(RHS&& rhs,
                               EventCheck&& ev,
                               ErrorEstimator&& err_est,
                               double s0,
                               const std::array<double,N>& y0,
                               double sMax,
                               const RKOpts& opt)
{
  IdentityProjector proj{};
  return rk4_adaptive<N>(rhs, ev, err_est, s0, y0, sMax, opt, proj);
}

// ---------- adaptive RK4 with default error estimator + PROJECTOR ----------
template<size_t N, class RHS, class EventCheck, class Projector>
inline RKResult<N> rk4_adaptive(RHS&& rhs,
                               EventCheck&& ev,
                               double s0,
                               const std::array<double,N>& y0,
                               double sMax,
                               const RKOpts& opt,
                               Projector&& proj)
{
  auto err_est = [&](const std::array<double,N>& y,
                     const std::array<double,N>& y_full,
                     const std::array<double,N>& y_2half,
                     const RKOpts& opt_in) -> double
  {
    return default_err_inf_raw<N>(y, y_full, y_2half, opt_in);
  };

  return rk4_adaptive<N>(rhs, ev, err_est, s0, y0, sMax, opt, proj);
}

// ---------- adaptive RK4 with default error estimator (legacy, no projector) ----------
template<size_t N, class RHS, class EventCheck>
inline RKResult<N> rk4_adaptive(RHS&& rhs,
                               EventCheck&& ev,
                               double s0,
                               const std::array<double,N>& y0,
                               double sMax,
                               const RKOpts& opt)
{
  IdentityProjector proj{};
  return rk4_adaptive<N>(rhs, ev, s0, y0, sMax, opt, proj);
}

} // namespace core










