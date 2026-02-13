#include <iostream>
#include <vector>
#include <atomic>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <thread>
#include <iomanip>
#include <limits>

#ifdef SRB_USE_OPENMP
  #include <omp.h>
#endif

#include "singly_rotating_black_ring.h"

static constexpr double kPi = 3.14159265358979323846;

// ---- stb image I/O (implementation in THIS ONE file only) ----
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static inline const char* rk_reason_name(core::RKReason r) {
  switch (r) {
    case core::RKReason::Done:     return "Done(sMax)";
    case core::RKReason::Event:    return "Event";
    case core::RKReason::Hmin:     return "Hmin";
    case core::RKReason::MaxSteps: return "MaxSteps";
    case core::RKReason::NaN:      return "NaN";
    default:                       return "Unknown";
  }
}

static inline double deg2rad(double d) { return d * kPi / 180.0; }
static inline double clamp01(double x) { return std::min(1.0, std::max(0.0, x)); }
static inline double clamp(double x, double a, double b) { return std::min(b, std::max(a, x)); }

// ------------------------------
// Stable helpers for new chart:
//   x = tanh(chi)
//   y = -cosh(eta)
//   D = x - y = tanh(chi) + cosh(eta) = (1+tanh) + (cosh-1)
// ------------------------------
static inline double tanh_plus_one_stable(double chi) {
  if (chi >= 0.0) {
    const double e = std::exp(-2.0*chi);
    return 2.0 / (1.0 + e);
  } else {
    const double e = std::exp( 2.0*chi);
    return 2.0 * e / (1.0 + e);
  }
}

static inline double one_minus_tanh_stable(double chi) {
  if (chi >= 0.0) {
    const double e = std::exp(-2.0*chi);
    return 2.0 * e / (1.0 + e);
  } else {
    const double e = std::exp( 2.0*chi);
    return 2.0 / (1.0 + e);
  }
}

static inline double sech2_stable(double chi) {
  const double a = std::abs(chi);
  const double t = std::exp(-2.0*a);
  const double d = 1.0 + t;
  return 4.0 * t / (d*d);
}

static inline long double coshm1_stable(double eta) {
  const long double h = 0.5L*(long double)eta;
  const long double sh = std::sinh(h);
  return 2.0L*sh*sh;
}

static inline long double D_stable(double chi, double eta) {
  return (long double)tanh_plus_one_stable(chi) + coshm1_stable(eta);
}

static inline long double rho2_stable(const singly_rotating_black_ring::Params& P,
                                      double chi, double eta)
{
  const long double num = coshm1_stable(eta) + (long double)one_minus_tanh_stable(chi);
  const long double den = D_stable(chi, eta);
  if (!(den > 0.0L)) return std::numeric_limits<long double>::quiet_NaN();
  return (long double)(P.R*P.R) * num / den;
}

// Convert (chi,eta,phi,psi) -> hyperplane R^3 coords (x1,x2,vCoord) using stable D.
//   r1 = R*sech(chi)/D
//   r2 = R*sinh(eta)/D
static inline bool ring_to_cart3_hyperplane_chieta(
    const singly_rotating_black_ring::Params& P,
    double chi, double eta, double phi, double psi,
    double v3, double v4,
    double& X1, double& X2, double& Vc)
{
  if (!std::isfinite(chi) || !std::isfinite(eta) || eta < 0.0) return false;

  const long double D = D_stable(chi, eta);
  if (!(D > 0.0L)) return false;

  const long double invD = 1.0L / D;
  const long double R_over_D = (long double)P.R * invD;

  const long double sech = std::sqrt((long double)sech2_stable(chi));
  const long double sH   = std::sinh((long double)eta);

  const long double r1 = R_over_D * sech;
  const long double r2 = R_over_D * sH;

  // wrap only for trig evaluation (doesn't mutate state)
  const double psi_w = core::wrap_pi(psi);
  const double phi_w = core::wrap_pi(phi);

  const double x1 = (double)r2 * std::cos(psi_w);
  const double x2 = (double)r2 * std::sin(psi_w);

  const double x3 = (double)r1 * std::cos(phi_w);
  const double x4 = (double)r1 * std::sin(phi_w);

  X1 = x1;
  X2 = x2;
  Vc = x3*v3 + x4*v4;
  return std::isfinite(X1) && std::isfinite(X2) && std::isfinite(Vc);
}

// Outgoing direction in hyperplane R^3 = (x1, x2, vCoord) from (chi,eta,phi,psi) and derivatives.
static inline bool outgoing_dir_cart3_from_chieta_derivs(
    const singly_rotating_black_ring::Params& P,
    double chi, double eta, double phi, double psi,
    double dchi, double deta, double dphi, double dpsi,
    double v3, double v4,
    double& dX1, double& dX2, double& dV)
{
  if (!std::isfinite(chi) || !std::isfinite(eta) || eta < 0.0) return false;

  const long double D = D_stable(chi, eta);
  if (!(D > 0.0L)) return false;

  const long double invD = 1.0L / D;
  const long double R_over_D = (long double)P.R * invD;

  const double x     = std::tanh(chi);
  const double sech2 = sech2_stable(chi);
  const long double sech = std::sqrt((long double)sech2);

  const long double sH = std::sinh((long double)eta);
  const long double cH = 1.0L + coshm1_stable(eta);

  const double r1 = (double)(R_over_D * sech);
  const double r2 = (double)(R_over_D * sH);

  // dD = sech^2 * dchi + sinh * deta
  const long double dD = (long double)sech2 * (long double)dchi + sH * (long double)deta;
  const long double dD_over_D = dD * invD;

  // d(sech)/dchi = -sech*tanh
  const long double dsech = -(sech * (long double)x) * (long double)dchi;

  // dr1 = (R/D)*dsech - r1*(dD/D)
  const double dr1 = (double)(R_over_D * dsech - (long double)r1 * dD_over_D);

  // d(sinh)/deta = cosh
  const long double dsinh = cH * (long double)deta;

  // dr2 = (R/D)*dsinh - r2*(dD/D)
  const double dr2 = (double)(R_over_D * dsinh - (long double)r2 * dD_over_D);

  const double psi_w = core::wrap_pi(psi);
  const double phi_w = core::wrap_pi(phi);

  // x1 = r2 cos psi, x2 = r2 sin psi
  const double cpsi = std::cos(psi_w), spsi = std::sin(psi_w);
  dX1 = dr2*cpsi - r2*spsi*dpsi;
  dX2 = dr2*spsi + r2*cpsi*dpsi;

  // vCoord = r1*(v3 cos phi + v4 sin phi)
  const double cphi = std::cos(phi_w), sphi = std::sin(phi_w);
  const double a = v3*cphi + v4*sphi;
  const double da_dphi = -v3*sphi + v4*cphi;

  dV = dr1*a + r1*da_dphi*dphi;

  return std::isfinite(dX1) && std::isfinite(dX2) && std::isfinite(dV);
}

static void debug_one_ray(
    const singly_rotating_black_ring::Params& P,
    const singly_rotating_black_ring::RenderOpts& R,
    const core::RKOpts& O,
    const singly_rotating_black_ring::Frame5& F,
    double v3, double v4,
    const core::Vec4& ncam,
    double sMax)
{
  using namespace singly_rotating_black_ring;

  Y7 y0{};
  RayConsts C{};
  if (!init_null_ray_from_frame(F, ncam, y0, C)) {
    std::cout << "debug_one_ray: init failed\n";
    return;
  }

  // IMPORTANT: debug uses the SAME integration path as render now.
  auto res = trace_ray(P, R, O, y0, C, sMax);

  const double chi_f = res.y_end[1];
  const double eta_f = res.y_end[2];
  const double x_f   = std::tanh(chi_f);
  const double y_f   = -std::cosh(eta_f);

  const long double Df = D_stable(chi_f, eta_f);
  const long double rho2f = rho2_stable(P, chi_f, eta_f);

  std::cout << "\n--- DEBUG ONE RAY ---\n";
  std::cout << "event=" << res.event << "  reason=" << rk_reason_name(res.reason)
            << "  n_accept=" << res.n_accept << "  n_reject=" << res.n_reject << "\n";
  std::cout << "s_end=" << res.s_end << "  sMax=" << sMax << "\n";
  std::cout << "final: chi=" << chi_f << " eta=" << eta_f
            << "   (x=" << x_f << " y=" << y_f << ")"
            << "  D=" << (double)Df << "  rho^2=" << (double)rho2f << "\n";
  std::cout << "r_escape^2=" << (R.r_escape * R.r_escape) << "\n";
  std::cout << "---------------------\n";
}

// Bilinear sampler for equirect RGB panorama.
static inline void sample_panorama_rgb_bilinear(
    const unsigned char* pano, int Wp, int Hp,
    double u, double v,
    unsigned char outRGB[3])
{
  if (!pano || Wp <= 0 || Hp <= 0) { outRGB[0]=outRGB[1]=outRGB[2]=255; return; }

  u = u - std::floor(u);
  v = clamp01(v);

  double x = u * (double)Wp;
  double y = v * (double)Hp;

  int x0 = (int)std::floor(x) % Wp;
  int y0 = (int)std::floor(y);
  if (y0 < 0) y0 = 0;
  if (y0 > Hp-1) y0 = Hp-1;

  int x1 = (x0 + 1) % Wp;
  int y1 = std::min(y0 + 1, Hp - 1);

  double tx = x - std::floor(x);
  double ty = y - std::floor(y);

  auto pix = [&](int xi, int yi, int c)->double {
    return (double)pano[3*(yi*Wp + xi) + c];
  };

  for (int c=0;c<3;c++) {
    double c00 = pix(x0,y0,c);
    double c10 = pix(x1,y0,c);
    double c01 = pix(x0,y1,c);
    double c11 = pix(x1,y1,c);

    double c0 = c00*(1.0-tx) + c10*tx;
    double c1 = c01*(1.0-tx) + c11*tx;
    double cf = c0*(1.0-ty) + c1*ty;

    int iv = (int)std::lround(cf);
    if (iv < 0) iv = 0;
    if (iv > 255) iv = 255;
    outRGB[c] = (unsigned char)iv;
  }
}

// Non-white fallback (debug-friendly).
static inline void sample_fallback_sky(double dx, double dy, double dz, unsigned char outRGB[3]) {
  (void)dx; (void)dy;
  double t = clamp01(0.5*(dz + 1.0));
  double r = (1.0 - t)*0.55 + t*0.95;
  double g = (1.0 - t)*0.70 + t*0.97;
  double b = (1.0 - t)*1.00 + t*1.00;
  outRGB[0] = (unsigned char)std::lround(255.0 * clamp01(r));
  outRGB[1] = (unsigned char)std::lround(255.0 * clamp01(g));
  outRGB[2] = (unsigned char)std::lround(255.0 * clamp01(b));
}

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cout.tie(nullptr);

  using namespace singly_rotating_black_ring;
  using core::Vec3; using core::Vec4;

  // ---- Load panorama ----
  const char* panoPath = (argc > 1) ? argv[1] : "panorama.jpg";
  int panoW=0, panoH=0, panoC=0;
  unsigned char* panoData = stbi_load(panoPath, &panoW, &panoH, &panoC, 3);

  if (!panoData) {
    std::cerr << "Warning: could not load panorama '" << panoPath
              << "' (" << stbi_failure_reason() << "). Using fallback sky.\n";
  } else {
    std::cout << "Loaded panorama " << panoW << "x" << panoH << " from " << panoPath << "\n";
    unsigned char test[3];
    sample_panorama_rgb_bilinear(panoData, panoW, panoH, 0.5, 0.5, test);
    std::cout << "Panorama sample @ (0.5,0.5): "
              << int(test[0]) << "," << int(test[1]) << "," << int(test[2]) << "\n";
  }

  // ---- Params (balanced) ----
  Params P;
  P.R = 10.0;
  P.nu = 0.4;
  P.lambda = 2*P.nu/(1 + P.nu*P.nu);

  // ---- Camera ----
  Vec4 v{}; v[0]=0; v[1]=0; v[2]=1; v[3]=0;   // v-axis in x3x4 plane
  Vec3 pos{}; pos[0]=1; pos[1]=1; pos[2]=60;  // your test position
  Vec3 up{};  up[0]=0; up[1]=0; up[2]=1;

  Frame5 F = build_camera_frame(P, pos, up, v);

  // Normalize v-hat in x3x4 plane for projecting x3,x4 -> vCoord
  double v3 = v[2], v4 = v[3];
  double nv = std::sqrt(v3*v3 + v4*v4);
  if (nv <= 0) { v3 = 1.0; v4 = 0.0; nv = 1.0; }
  v3 /= nv; v4 /= nv;

  RenderOpts R;
  R.horizon_eps = 1e-5;

  double cam_r = std::sqrt(F.p4[0]*F.p4[0] + F.p4[1]*F.p4[1] + F.p4[2]*F.p4[2] + F.p4[3]*F.p4[3]);
  R.r_escape = 5.0 * cam_r;

  core::RKOpts O;
  O.rtol = 1e-6;
  O.atol = 1e-6;
  O.h0   = 0.5;
  O.hmin = 1e-12;              // critical: do NOT use 1e-6 here
  O.hmax = 20.0;
  O.max_steps = 400000;
  O.force_accept_hmin = true;  // critical: don't paint the world purple at hmin

  // ---- Render ----
  const int W = 320;
  const int H = 240;
  const double FOV_h = 70.0;
  const double FOV_v = 50.0;
  const double tan_h = std::tan(0.5*deg2rad(FOV_h));
  const double tan_v = std::tan(0.5*deg2rad(FOV_v));

  const double sMax = 500.0 * R.r_escape;

  std::vector<unsigned char> rgb(W*H*3, 255);

  // Center pixel ray: forward=1
  core::Vec4 ncam{};
  ncam[0]=0.0; ncam[1]=0.0; ncam[2]=1.0; ncam[3]=0.0;

  double sMax_debug = 1e6;
  debug_one_ray(P, R, O, F, v3, v4, ncam, sMax_debug);

#ifdef SRB_USE_OPENMP
  const int nWorkers = omp_get_max_threads();
#else
  const int nWorkers = 1;
#endif

  const int N = W*H;
  std::cout << "Rendering " << W << "x" << H << " (" << N << " rays) using "
            << nWorkers << " worker(s)\n";

  // ---- Fast progress: background thread + amortized atomics ----
  std::atomic<int> done{0};
  std::atomic<bool> stopProgress{false};
  auto t0 = std::chrono::steady_clock::now();

  std::thread progressThread([&](){
    int last = -1;
    while (!stopProgress.load(std::memory_order_relaxed)) {
      int k = done.load(std::memory_order_relaxed);
      if (k != last) {
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double pct = 100.0 * double(k) / double(N);
        std::cout << "\r  " << std::setw(6) << k << " / " << N
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                  << "  elapsed " << std::setprecision(1) << sec << "s" << std::flush;
        last = k;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });

  // ---- Stats counters (to explain purple) ----
  std::atomic<long long> cnt_horizon{0}, cnt_escape{0}, cnt_invalid{0};
  std::atomic<long long> cnt_done{0}, cnt_hmin{0}, cnt_nan{0}, cnt_maxsteps{0}, cnt_other{0};

  const int flushEvery = 256;
  const int chunk = 16;

#ifdef SRB_USE_OPENMP
  #pragma omp parallel
  {
    int localDone = 0;
    long long lh=0, le=0, li=0, ldone=0, lhmin=0, lnan=0, lms=0, lother=0;

    #pragma omp for schedule(dynamic, chunk)
    for (int idx=0; idx<N; ++idx) {
      int j = idx / W;
      int i = idx - j*W;

      double sy = 1.0 - 2.0*((j + 0.5)/double(H));
      double vScreen = tan_v * sy;

      double sx = 2.0*((i + 0.5)/double(W)) - 1.0;
      double uScreen = tan_h * sx;

      Vec4 ncam{};
      ncam[0]=uScreen; ncam[1]=vScreen; ncam[2]=1.0; ncam[3]=0.0;

      Y7 y0{};
      RayConsts C{};
      bool ok = init_null_ray_from_frame(F, ncam, y0, C);

      unsigned char pixRGB[3] = {255,255,255};

      if (ok) {
        auto res = trace_ray(P, R, O, y0, C, sMax);

        if (res.event == 1) {
          lh++;
          pixRGB[0]=pixRGB[1]=pixRGB[2]=0;
        } else if (res.event == 2) {
          le++;
          // ONLY sample pano for true escape
          Y7 dyds{};
          rhs(P, C, res.s_end, res.y_end, dyds);

          const double chi = res.y_end[1];
          const double eta = res.y_end[2];
          const double phi = res.y_end[3];
          const double psi = res.y_end[4];

          const double dchi = dyds[1];
          const double deta = dyds[2];
          const double dphi = dyds[3];
          const double dpsi = dyds[4];

          double dX1, dX2, dV;
          if (outgoing_dir_cart3_from_chieta_derivs(P, chi, eta, phi, psi, dchi, deta, dphi, dpsi, v3, v4, dX1, dX2, dV)) {
            double nn = std::sqrt(dX1*dX1 + dX2*dX2 + dV*dV);
            if (nn > 0.0) {
              double ux = dX1/nn, uy = dX2/nn, uz = dV/nn;

              double az = std::atan2(uy, ux);
              double theta = std::acos(clamp(uz, -1.0, 1.0));

              double uTex = (az + kPi) / (2.0*kPi);
              double vTex = theta / kPi;

              if (panoData) sample_panorama_rgb_bilinear(panoData, panoW, panoH, uTex, vTex, pixRGB);
              else          sample_fallback_sky(ux, uy, uz, pixRGB);
            }
          }
        } else if (res.event == 3) {
          li++;
          pixRGB[0]=255; pixRGB[1]=0; pixRGB[2]=255;
        } else {
          // event==0: classify by RKReason
          switch (res.reason) {
            case core::RKReason::Done:     ldone++; break;
            case core::RKReason::Hmin:     lhmin++; break;
            case core::RKReason::NaN:      lnan++;  break;
            case core::RKReason::MaxSteps: lms++;   break;
            default:                       lother++; break;
          }
          pixRGB[0]=255; pixRGB[1]=0; pixRGB[2]=255;
        }
      }

      rgb[3*idx+0] = pixRGB[0];
      rgb[3*idx+1] = pixRGB[1];
      rgb[3*idx+2] = pixRGB[2];

      localDone++;
      if (localDone == flushEvery) {
        done.fetch_add(localDone, std::memory_order_relaxed);
        localDone = 0;
      }
    }

    if (localDone > 0) done.fetch_add(localDone, std::memory_order_relaxed);

    cnt_horizon.fetch_add(lh, std::memory_order_relaxed);
    cnt_escape .fetch_add(le, std::memory_order_relaxed);
    cnt_invalid.fetch_add(li, std::memory_order_relaxed);
    cnt_done   .fetch_add(ldone, std::memory_order_relaxed);
    cnt_hmin   .fetch_add(lhmin, std::memory_order_relaxed);
    cnt_nan    .fetch_add(lnan, std::memory_order_relaxed);
    cnt_maxsteps.fetch_add(lms, std::memory_order_relaxed);
    cnt_other  .fetch_add(lother, std::memory_order_relaxed);
  }
#else
  for (int idx=0; idx<N; ++idx) {
    rgb[3*idx+0]=255; rgb[3*idx+1]=0; rgb[3*idx+2]=255;
    done.fetch_add(1, std::memory_order_relaxed);
  }
#endif

  done.store(N, std::memory_order_relaxed);
  stopProgress.store(true, std::memory_order_relaxed);
  progressThread.join();
  std::cout << "\n";

  if (panoData) stbi_image_free(panoData);

  if (!stbi_write_png("blackring_env.png", W, H, 3, rgb.data(), W * 3)) {
    std::cerr << "Failed to write blackring_env.png\n";
    return 1;
  }
  std::cout << "Saved blackring_env.png\n";

  // Print stats to explain purple
  std::cout << "Ray stats:\n";
  std::cout << "  horizon(event=1): " << cnt_horizon.load() << "\n";
  std::cout << "  escape (event=2): " << cnt_escape.load()  << "\n";
  std::cout << "  invalid(event=3): " << cnt_invalid.load() << "\n";
  std::cout << "  event=0 + reason Done(sMax): " << cnt_done.load() << "\n";
  std::cout << "  event=0 + reason Hmin:       " << cnt_hmin.load() << "\n";
  std::cout << "  event=0 + reason NaN:        " << cnt_nan.load() << "\n";
  std::cout << "  event=0 + reason MaxSteps:   " << cnt_maxsteps.load() << "\n";
  std::cout << "  event=0 + reason Other:      " << cnt_other.load() << "\n";

  return 0;
}










