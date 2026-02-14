#include <iostream>
#include <vector>
#include <atomic>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <thread>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <string>

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
// Stable helpers for (x,y) chart
// ------------------------------
static inline long double D_stable_xy(double x, double y) {
  return (1.0L + (long double)x) + (-1.0L - (long double)y);
}
static inline long double omx2_stable_xy(double x) {
  const long double xl = (long double)x;
  return (1.0L - xl) * (1.0L + xl);
}
static inline long double y2m1_stable_xy(double y) {
  const long double yl = (long double)y;
  return (yl - 1.0L) * (yl + 1.0L);
}
static inline long double rho2_stable_xy(const singly_rotating_black_ring::Params& P,
                                        double x, double y)
{
  const long double D = D_stable_xy(x, y);
  if (!(D > 0.0L)) return std::numeric_limits<long double>::quiet_NaN();
  const long double num = -(long double)y - (long double)x;
  return (long double)(P.R*P.R) * num / D;
}

// Outgoing direction in hyperplane R^3 = (x1, x2, vCoord) from (x,y,phi,psi) and derivatives.
static inline bool outgoing_dir_cart3_from_xy_derivs(
    const singly_rotating_black_ring::Params& P,
    double x, double y, double phi, double psi,
    double dx, double dy, double dphi, double dpsi,
    double v3, double v4,
    double& dX1, double& dX2, double& dV)
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(phi) || !std::isfinite(psi)) return false;
  if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dphi) || !std::isfinite(dpsi)) return false;
  if (!(x > -1.0 && x < 1.0)) return false;
  if (!(y <= -1.0)) return false;

  const long double D = D_stable_xy(x, y);
  if (!(D > 0.0L)) return false;

  long double omx2 = omx2_stable_xy(x);
  long double y2m1 = y2m1_stable_xy(y);
  if (omx2 < 0.0L) omx2 = 0.0L;
  if (y2m1 < 0.0L) y2m1 = 0.0L;

  const long double invD = 1.0L / D;
  const long double R_over_D = (long double)P.R * invD;

  const long double s1 = std::sqrt(omx2);
  const long double s2 = std::sqrt(y2m1);

  const double r1 = (double)(R_over_D * s1);
  const double r2 = (double)(R_over_D * s2);

  // dD = dx - dy
  const long double dD = (long double)dx - (long double)dy;
  const long double dD_over_D = dD * invD;

  // ds1 = d sqrt(1-x^2) = (-x/s1) dx
  // ds2 = d sqrt(y^2-1) = (y/s2) dy
  constexpr long double S_EPS = 1e-300L;
  const long double invs1 = 1.0L / std::max(s1, S_EPS);
  const long double invs2 = 1.0L / std::max(s2, S_EPS);

  const long double ds1 = (-(long double)x) * (long double)dx * invs1;
  const long double ds2 = ( (long double)y) * (long double)dy * invs2;

  // dr = (R/D)*ds - r*(dD/D)
  const double dr1 = (double)(R_over_D * ds1 - (long double)r1 * dD_over_D);
  const double dr2 = (double)(R_over_D * ds2 - (long double)r2 * dD_over_D);

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

static void debug_one_ray_dump(
    const singly_rotating_black_ring::Params& P,
    const singly_rotating_black_ring::RenderOpts& R,
    const core::RKOpts& O,
    const singly_rotating_black_ring::Frame5& F,
    const core::Vec4& ncam,
    double sMax,
    const char* csv_path)
{
  using namespace singly_rotating_black_ring;

  Y7 y0{};
  RayConsts C{};
  if (!init_null_ray_from_frame(F, ncam, y0, C)) {
    std::cout << "debug_one_ray: init failed\n";
    return;
  }

  auto res = trace_ray_dump_csv(P, R, O, y0, C, sMax, csv_path);
  std::cout << "Wrote trajectory CSV: " << csv_path << "\n";

  const double x_f = res.y_end[1];
  const double y_f = res.y_end[2];

  const long double Df = D_stable_xy(x_f, y_f);
  const long double rho2f = rho2_stable_xy(P, x_f, y_f);

  std::cout << "\n--- DEBUG ONE RAY ---\n";
  std::cout << "rescale_pow=" << R.rescale_pow << "\n";
  std::cout << "event=" << res.event << "  reason=" << rk_reason_name(res.reason)
            << "  n_accept=" << res.n_accept << "  n_reject=" << res.n_reject << "\n";
  std::cout << "s_end=" << res.s_end << "  sMax=" << sMax << "\n";
  std::cout << "final: x=" << x_f << " y=" << y_f
            << "  D=" << (double)Df << "  rho^2=" << (double)rho2f << "\n";
  std::cout << "r_escape^2=" << (R.r_escape * R.r_escape) << "\n";
  std::cout << "---------------------\n";
}

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cout.tie(nullptr);

  using namespace singly_rotating_black_ring;
  using core::Vec3; using core::Vec4;

  // ---- Load panorama ----
  const char* panoPath = (argc > 1) ? argv[1] : "panorama.jpg";

  int rescale_pow = 0;
  if (argc > 2) rescale_pow = std::clamp(std::atoi(argv[2]), 0, 2);

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
  Vec4 v{}; v[0]=0; v[1]=0; v[2]=1; v[3]=0;
  Vec3 pos{}; pos[0]=1; pos[1]=1; pos[2]=60;
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
  R.rescale_pow = rescale_pow;

  core::RKOpts O;
  O.rtol = 1e-6;
  O.atol = 1e-6;
  O.h0   = 0.5;
  O.hmin = 1e-12;
  O.hmax = 20.0;
  O.max_steps = 400000;
  O.force_accept_hmin = true;

  // ---- Render ----
  const int W = 320;
  const int H = 240;
  const double FOV_h = 70.0;
  const double FOV_v = 50.0;
  const double tan_h = std::tan(0.5*deg2rad(FOV_h));
  const double tan_v = std::tan(0.5*deg2rad(FOV_v));

  int dbg_i = W/2;
  int dbg_j = H/2;
  if (argc > 3) dbg_i = std::atoi(argv[3]);
  if (argc > 4) dbg_j = std::atoi(argv[4]);
  dbg_i = std::clamp(dbg_i, 0, W-1);
  dbg_j = std::clamp(dbg_j, 0, H-1);

  const double sMax = 500.0 * R.r_escape;

  std::vector<unsigned char> rgb(W*H*3, 255);

  // ---- DEBUG RAY (selectable) ----
  {
    double sy = 1.0 - 2.0*((dbg_j + 0.5)/double(H));
    double vScreen = tan_v * sy;

    double sx = 2.0*((dbg_i + 0.5)/double(W)) - 1.0;
    double uScreen = tan_h * sx;

    Vec4 ncam{};
    ncam[0]=uScreen; ncam[1]=vScreen; ncam[2]=1.0; ncam[3]=0.0;

    std::string csv = "debug_ray_p" + std::to_string(R.rescale_pow)
                    + "_i" + std::to_string(dbg_i)
                    + "_j" + std::to_string(dbg_j) + ".csv";

    double sMax_debug = 1e6;
    debug_one_ray_dump(P, R, O, F, ncam, sMax_debug, csv.c_str());
  }

#ifdef SRB_USE_OPENMP
  const int nWorkers = omp_get_max_threads();
#else
  const int nWorkers = 1;
#endif

  const int N = W*H;
  std::cout << "Rendering " << W << "x" << H << " (" << N << " rays) using "
            << nWorkers << " worker(s)\n";

  // ---- Fast progress ----
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

  // ---- Stats ----
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

          const double x   = res.y_end[1];
          const double y   = res.y_end[2];
          const double phi = res.y_end[3];
          const double psi = res.y_end[4];

          const double dx   = dyds[1];
          const double dy   = dyds[2];
          const double dphi = dyds[3];
          const double dpsi = dyds[4];

          double dX1, dX2, dV;
          if (outgoing_dir_cart3_from_xy_derivs(P, x, y, phi, psi, dx, dy, dphi, dpsi, v3, v4, dX1, dX2, dV)) {
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

  char outName[128];
  std::snprintf(outName, sizeof(outName), "blackring_env_p%d.png", rescale_pow);

  if (!stbi_write_png(outName, W, H, 3, rgb.data(), W * 3)) {
    std::cerr << "Failed to write " << outName << "\n";
    return 1;
  }
  std::cout << "Saved " << outName << "\n";

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











