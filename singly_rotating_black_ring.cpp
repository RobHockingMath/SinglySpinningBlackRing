#include "singly_rotating_black_ring.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <array>
#include <fstream>
#include <iomanip>

namespace singly_rotating_black_ring {

static inline double sqr(double x) { return x*x; }

static inline double inner_g(const Vec5& a, const Vec5& b, const Mat5& g) {
  // a^T g b
  double s = 0.0;
  for (int i=0;i<5;i++) {
    double row = 0.0;
    for (int j=0;j<5;j++) row += g(i,j)*b[j];
    s += a[i]*row;
  }
  return s;
}

static inline Vec5 mat5_mul(const Mat5& g, const Vec5& v) {
  return core::mul(g, v);
}

// -------- stable primitives in (x,y) --------
static inline long double D_stable_xy(double x, double y) {
  // x - y = (1+x) + (-1-y) (more stable near (-1,-1))
  return (1.0L + (long double)x) + (-1.0L - (long double)y);
}
static inline long double omx2_stable_xy(double x) {
  // 1 - x^2 = (1-x)(1+x)
  const long double xl = (long double)x;
  return (1.0L - xl) * (1.0L + xl);
}
static inline long double y2m1_stable_xy(double y) {
  // y^2 - 1 = (y-1)(y+1)
  const long double yl = (long double)y;
  return (yl - 1.0L) * (yl + 1.0L);
}

struct RingPointXY {
  double x, y, phi, psi;
  double r1, r2, Sigma, A;
};

// --- original cart4 -> ring (x,y,phi,psi) map ---
static RingPointXY cart4_to_ring_xy(const Vec4& p4, double R) {
  const double x1 = p4[0], x2 = p4[1], x3 = p4[2], x4 = p4[3];
  const double r2 = std::sqrt(sqr(x1) + sqr(x2));
  const double r1 = std::sqrt(sqr(x3) + sqr(x4));
  if (r2 < 1e-14) throw std::runtime_error("r2 ~ 0 (psi undefined)");
  if (r1 < 1e-14) throw std::runtime_error("r1 ~ 0 (phi undefined)");

  const double psi = std::atan2(x2, x1);
  const double phi = std::atan2(x4, x3);

  const double A = r1*r1 + r2*r2 + R*R;
  double Sigma2 = A*A - 4.0*R*R*r2*r2;
  Sigma2 = std::max(0.0, Sigma2);
  const double Sigma = std::sqrt(Sigma2);

  const double y = -(R*R + r1*r1 + r2*r2) / Sigma;  // = -A/Sigma
  const double x =  (R*R - r1*r1 - r2*r2) / Sigma;

  return {x,y,phi,psi,r1,r2,Sigma,A};
}

static Mat4 cart4_to_ring_xy_jacobian(const Vec4& p4, double R,
                                      double r1, double r2, double Sigma)
{
  // J = d(x,y,phi,psi)/d(x1,x2,x3,x4)
  const double x1 = p4[0], x2 = p4[1], x3 = p4[2], x4 = p4[3];

  if (r1 < 1e-14 || r2 < 1e-14) throw std::runtime_error("Jacobian ill-conditioned near axis");

  const double Sig3 = Sigma*Sigma*Sigma;

  const double dx_dr1 = -4*R*R*r1*(r1*r1 - r2*r2 + R*R)/Sig3;
  const double dx_dr2 = -8*R*R*(r1*r1)*r2              /Sig3;

  const double dy_dr1 =  8*R*R*r1*(r2*r2)              /Sig3;
  const double dy_dr2 =  4*R*R*r2*(r2*r2 - r1*r1 - R*R)/Sig3;

  Vec4 dr1{}; dr1[0]=0; dr1[1]=0; dr1[2]=x3/r1; dr1[3]=x4/r1;
  Vec4 dr2{}; dr2[0]=x1/r2; dr2[1]=x2/r2; dr2[2]=0; dr2[3]=0;

  Vec4 dphi{}; dphi[0]=0; dphi[1]=0; dphi[2]= -x4/(r1*r1); dphi[3]= x3/(r1*r1);
  Vec4 dpsi{}; dpsi[0]= -x2/(r2*r2); dpsi[1]= x1/(r2*r2); dpsi[2]=0; dpsi[3]=0;

  Vec4 dx{}, dy{};
  for (int i=0;i<4;i++) {
    dx[i] = dx_dr1*dr1[i] + dx_dr2*dr2[i];
    dy[i] = dy_dr1*dr1[i] + dy_dr2*dr2[i];
  }

  Mat4 J = Mat4::zero();
  for (int c=0;c<4;c++) { J(0,c)=dx[c]; J(1,c)=dy[c]; J(2,c)=dphi[c]; J(3,c)=dpsi[c]; }
  return J;
}

static inline double compute_Cc_cached(double lambda, double nu)
{
  thread_local double cache_la = std::numeric_limits<double>::quiet_NaN();
  thread_local double cache_nu = std::numeric_limits<double>::quiet_NaN();
  thread_local double cache_Cc = 0.0;
  if (!(lambda == cache_la && nu == cache_nu)) {
    cache_la = lambda;
    cache_nu = nu;
    cache_Cc = std::sqrt(lambda*(lambda-nu) * (1.0+lambda)/(1.0-lambda));
  }
  return cache_Cc;
}

// Build covariant metric at a point by constructing contravariant pieces (stable)
// and inverting the 2x2 (t,psi) block. This avoids removable singularities at Fy=0.
static Mat5 metric_srb_xy_cov_stable(double x, double y, const Params& P)
{
  constexpr long double OMx2_MIN = 1e-300L;
  constexpr long double Y2M1_MIN = 1e-300L;
  constexpr double A_MIN = 1e-15;

  const double R  = P.R;
  const double la = P.lambda;
  const double nu = P.nu;

  const long double Dld = D_stable_xy(x,y);
  if (!(Dld > 0.0L)) throw std::runtime_error("D<=0 in metric");

  const long double D2 = Dld*Dld;

  const double Fx = 1.0 + la*x;
  const double Fy = 1.0 + la*y;
  if (!std::isfinite(Fx) || Fx == 0.0 || !std::isfinite(Fy)) throw std::runtime_error("bad F(x/y)");

  const long double omx2 = std::max(omx2_stable_xy(x), OMx2_MIN);
  const long double y2m1 = std::max(y2m1_stable_xy(y), Y2M1_MIN);

  const double onepnx = 1.0 + nu*x;
  const double A = 1.0 + nu*y;
  const double A_safe = (std::abs(A) < A_MIN) ? (A >= 0.0 ? A_MIN : -A_MIN) : A;

  const long double Gx = omx2 * (long double)onepnx;
  const long double YA = y2m1 * (long double)A_safe; // (y^2-1)A  = -G(y) in exterior

  const long double R2 = (long double)R*(long double)R;

  const long double invP = D2 / (R2 * (long double)Fx); // D^2/(R^2 Fx)

  // g^{xx}, g^{yy}, g^{phi phi}
  const long double gxx   = Gx * invP;
  const long double gyy   = YA * invP;
  const long double gphph = D2 / (R2 * Gx);

  // t-psi block
  const double Cc = compute_Cc_cached(la, nu);

  const long double B = 1.0L - (long double)y; // 1 - y (>0 in exterior)

  const long double gtps  = -(long double)Cc * D2 / ((long double)R * (long double)Fx * B * (long double)A_safe);
  const long double gpsps = (long double)Fy * D2 / (R2 * (long double)Fx * (long double)A_safe * y2m1);

  // g^{tt} (Fy-free rational form, from your earlier stabilized code)
  const long double c = -(long double)y; // cosh-like
  const long double xl = (long double)x;
  const long double lal = (long double)la;
  const long double nul = (long double)nu;

  const long double Npoly =
      -c*c*lal*lal + c*c*lal*nul - c*c*lal + c*c*nul
    + c*lal*lal*nul*xl*xl - 2.0L*c*lal*lal*xl + c*lal*lal
    - c*lal*nul*xl*xl + 4.0L*c*lal*nul*xl - c*lal*nul - 2.0L*c*lal*xl + c*nul - c
    + lal*lal*nul*xl*xl - 2.0L*lal*lal*xl*xl + 2.0L*lal*lal*xl
    + lal*nul*xl*xl - 2.0L*lal*xl + lal - 1.0L;

  const long double Den_tt =
      (c + 1.0L) * (lal - 1.0L) * (c*nul - 1.0L) * (lal*xl + 1.0L);

  if (Den_tt == 0.0L) throw std::runtime_error("Den_tt=0 in metric");
  const long double gtt = Npoly / Den_tt;

  // Invert (t,psi) 2x2 block
  const long double det = gtt*gpsps - gtps*gtps;
  if (det == 0.0L) throw std::runtime_error("det(t,psi)=0 in metric");

  const long double g_tt   =  gpsps / det;
  const long double g_tpsi = -gtps  / det;
  const long double g_psipsi = gtt  / det;

  Mat5 g = Mat5::zero();
  // indices: 0=t,1=x,2=y,3=phi,4=psi
  g(0,0) = (double)g_tt;
  g(0,4) = (double)g_tpsi;
  g(4,0) = (double)g_tpsi;
  g(4,4) = (double)g_psipsi;

  g(1,1) = (double)(1.0L / gxx);
  g(2,2) = (double)(1.0L / gyy);
  g(3,3) = (double)(1.0L / gphph);

  return g;
}

static Vec5 lnr_u(const Mat5& g)
{
  const double g_tt = g(0,0);
  const double g_tpsi = g(0,4);
  const double g_psipsi = g(4,4);
  if (std::abs(g_psipsi) < 1e-18) throw std::runtime_error("g_psipsi ~ 0 near axis");

  const double Omega = -g_tpsi / g_psipsi;
  const double alpha2 = -(g_tt - (g_tpsi*g_tpsi)/g_psipsi);
  if (alpha2 <= 0) throw std::runtime_error("LNR alpha^2 <= 0 (not timelike here)");

  const double alpha = std::sqrt(alpha2);

  Vec5 u{};
  u[0] = 1.0/alpha;
  u[1] = 0.0;
  u[2] = 0.0;
  u[3] = 0.0;
  u[4] = Omega/alpha;

  const double uu = inner_g(u,u,g);
  const double s = 1.0/std::sqrt(std::max(1e-30, -uu));
  for (int i=0;i<5;i++) u[i] *= s;

  return u;
}

static void metric_gram_schmidt(const Mat5& g, const Vec5& u,
                                const std::array<Vec5,4>& V,
                                Mat5& E5_out)
{
  core::set_col(E5_out, 0, u);

  std::array<Vec5,4> E{};
  for (int i=0;i<4;i++) {
    Vec5 w = V[i];

    const double wu = inner_g(w,u,g);
    const double uu = inner_g(u,u,g);
    const double fac_u = wu/uu;
    for (int k=0;k<5;k++) w[k] -= fac_u * u[k];

    for (int j=0;j<i;j++) {
      const double wj = inner_g(w, E[j], g);
      for (int k=0;k<5;k++) w[k] -= wj * E[j][k];
    }

    const double n2 = inner_g(w,w,g);
    if (!(n2 > 1e-18)) throw std::runtime_error("Gram–Schmidt failed (degenerate basis)");

    const double invn = 1.0/std::sqrt(n2);
    for (int k=0;k<5;k++) w[k] *= invn;
    E[i] = w;
    core::set_col(E5_out, i+1, w);
  }
}

static void build_euclid_frame(const Vec3& pos, const Vec3& up, const Vec4& v,
                               Mat4& E4_out, Vec4& p4_out)
{
  const double tol = 1e-12;

  Vec4 v_eff = v;
  v_eff[0] = 0.0;
  v_eff[1] = 0.0;

  double nv = core::norm(v_eff);
  if (nv < tol) throw std::runtime_error("v provides no direction orthogonal to e1,e2");
  Vec4 vhat = core::mul(1.0/nv, v_eff);

  auto proj3 = [&](const Vec4& X)->core::Vec3 {
    core::Vec3 r{};
    r[0] = X[0];
    r[1] = X[1];
    r[2] = X[2]*vhat[2] + X[3]*vhat[3];
    return r;
  };
  auto lift4 = [&](const core::Vec3& x3)->Vec4 {
    Vec4 X{};
    X[0]=x3[0];
    X[1]=x3[1];
    X[2]=x3[2]*vhat[2];
    X[3]=x3[2]*vhat[3];
    return X;
  };

  Vec4 p4{};
  p4[0]=pos[0]; p4[1]=pos[1];
  p4[2]=pos[2]*vhat[2]; p4[3]=pos[2]*vhat[3];

  Vec4 up4{};
  up4[0]=up[0]; up4[1]=up[1];
  up4[2]=up[2]*vhat[2]; up4[3]=up[2]*vhat[3];

  core::Vec3 p3 = proj3(p4);
  core::Vec3 up3 = proj3(up4);

  core::Vec3 f3 = core::mul(-1.0, p3);
  double nf = core::norm(f3);
  if (nf < tol) throw std::runtime_error("camera too close to origin");
  f3 = core::mul(1.0/nf, f3);

  double ufdot = core::dot(up3, f3);
  core::Vec3 u3 = core::sub(up3, core::mul(ufdot, f3));
  double nu3 = core::norm(u3);
  if (nu3 < tol) {
    core::Vec3 cands[3] = { core::Vec3{{0,0,1}}, core::Vec3{{0,1,0}}, core::Vec3{{1,0,0}} };
    bool found=false;
    for (auto& c : cands) {
      core::Vec3 utry = core::sub(c, core::mul(core::dot(c,f3), f3));
      if (core::norm(utry) > tol) { u3=utry; found=true; break; }
    }
    if (!found) throw std::runtime_error("failed to build up vector");
    nu3 = core::norm(u3);
  }
  u3 = core::mul(1.0/nu3, u3);

  core::Vec3 r3 = core::cross(f3, u3);
  double nr = core::norm(r3);
  if (nr < tol) throw std::runtime_error("degenerate right vector");
  r3 = core::mul(1.0/nr, r3);

  u3 = core::cross(r3, f3);
  u3 = core::mul(1.0/core::norm(u3), u3);

  Vec4 right4 = lift4(r3);
  Vec4 up4b   = lift4(u3);
  Vec4 fwd4   = lift4(f3);

  Vec4 c1{}; c1[2]=1;
  Vec4 c2{}; c2[3]=1;
  auto proj_out_S = [&](const Vec4& c)->Vec4 {
    Vec4 w = c;
    w[0]=0; w[1]=0;
    double cv = w[2]*vhat[2] + w[3]*vhat[3];
    w[2] -= cv*vhat[2];
    w[3] -= cv*vhat[3];
    return w;
  };
  Vec4 w1 = proj_out_S(c1);
  Vec4 w2 = proj_out_S(c2);
  Vec4 w = (core::norm(w2) > core::norm(w1)) ? w2 : w1;
  double nw = core::norm(w);
  if (nw < tol) throw std::runtime_error("failed to build w");
  w = core::mul(1.0/nw, w);

  E4_out = Mat4::zero();
  core::set_col(E4_out, 0, right4);
  core::set_col(E4_out, 1, up4b);
  core::set_col(E4_out, 2, fwd4);
  core::set_col(E4_out, 3, w);

  p4_out = p4;
}

Frame5 build_camera_frame(const Params& P, const Vec3& pos, const Vec3& up, const Vec4& v)
{
  if (!(P.R > 0 && P.nu > 0 && P.lambda >= P.nu && P.lambda < 1.0))
    throw std::runtime_error("Params invalid (need R>0 and 0<nu<=lambda<1)");

  Mat4 E4;
  Vec4 p4;
  build_euclid_frame(pos, up, v, E4, p4);

  RingPointXY rp = cart4_to_ring_xy(p4, P.R);

  constexpr double eps = 1e-15;
  double x = std::clamp(rp.x, -1.0 + eps, 1.0 - eps);
  double y = std::min(rp.y, -1.0 - eps);

  Vec5 q{};
  q[0]=0.0;
  q[1]=x;
  q[2]=y;
  q[3]=rp.phi;
  q[4]=rp.psi;

  Mat4 Jxy = cart4_to_ring_xy_jacobian(p4, P.R, rp.r1, rp.r2, rp.Sigma);

  std::array<Vec5,4> V{};
  for (int k=0;k<4;k++) {
    Vec4 ek = core::col(E4, k);
    Vec4 qv = core::mul(Jxy, ek); // [dx,dy,dphi,dpsi]
    Vec5 vk{};
    vk[0]=0.0;
    vk[1]=qv[0];
    vk[2]=qv[1];
    vk[3]=qv[2];
    vk[4]=qv[3];
    V[k]=vk;
  }

  Mat5 g = metric_srb_xy_cov_stable(x, y, P);
  Vec5 u5 = lnr_u(g);

  Mat5 E5 = Mat5::zero();
  metric_gram_schmidt(g, u5, V, E5);

  Frame5 F{};
  F.E5 = E5;
  F.g  = g;
  F.q  = q;
  F.p4 = p4;
  return F;
}

// ==============================
// normalize null scale so E = -p_t = 1
// ==============================
bool init_null_ray_from_frame(const Frame5& F, const Vec4& n_cam, Y7& y0, RayConsts& cst)
{
  double nn = std::sqrt(std::max(0.0, core::dot(n_cam, n_cam)));
  if (nn <= 0) return false;

  Vec4 n = core::mul(1.0/nn, n_cam);

  Vec5 u  = core::col(F.E5, 0);
  Vec5 e1 = core::col(F.E5, 1);
  Vec5 e2 = core::col(F.E5, 2);
  Vec5 e3 = core::col(F.E5, 3);
  Vec5 e4 = core::col(F.E5, 4);

  Vec5 k{};
  for (int i=0;i<5;i++) {
    k[i] = u[i] + n[0]*e1[i] + n[1]*e2[i] + n[2]*e3[i] + n[3]*e4[i];
  }

  Vec5 p = mat5_mul(F.g, k); // covariant

  double E0 = -p[0];
  if (!std::isfinite(E0) || E0 == 0.0) return false;

  if (E0 < 0.0) {
    for (int i=0;i<5;i++) p[i] = -p[i];
    E0 = -E0;
  }

  const double scale = 1.0 / E0;
  for (int i=0;i<5;i++) p[i] *= scale;

  cst.E   = 1.0;
  cst.Phi = p[3];
  cst.Psi = p[4];

  y0[0] = F.q[0];
  y0[1] = F.q[1]; // x
  y0[2] = F.q[2]; // y
  y0[3] = F.q[3]; // phi
  y0[4] = F.q[4]; // psi
  y0[5] = p[1];   // p_x
  y0[6] = p[2];   // p_y
  return true;
}

// =====================================================
// RHS in (x,y), Hamiltonian first-order form, UNRESCALED (p=0).
// State: [t, x, y, phi, psi, p_x, p_y]
// =====================================================
void rhs(const Params& P, const RayConsts& C, double /*s*/, const Y7& Y, Y7& dY)
{
  // The only "safety clamps" are for TRUE singular denominators (axis/horizon/asymptote).
  constexpr long double OMx2_MIN = 1e-300L;
  constexpr long double Y2M1_MIN = 1e-300L;
  constexpr double A_MIN = 1e-15;

  const double x  = Y[1];
  const double y  = Y[2];
  const double px = Y[5];
  const double py = Y[6];

  const double R  = P.R;
  const double la = P.lambda;
  const double nu = P.nu;

  const double E   = C.E;   // should be 1
  const double Phi = C.Phi;
  const double Psi = C.Psi;

  // Domain sanity
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(px) || !std::isfinite(py) ||
      !(x > -1.0 && x < 1.0) || !(y <= -1.0)) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double &v : dY) v = nan;
    return;
  }

  const long double Dld = D_stable_xy(x,y);
  if (!(Dld > 0.0L) || !std::isfinite((double)Dld)) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double &v : dY) v = nan;
    return;
  }
  const long double D2 = Dld*Dld;

  const double Fx = 1.0 + la*x;
  const double Fy = 1.0 + la*y;
  if (!std::isfinite(Fx) || Fx == 0.0 || !std::isfinite(Fy)) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double &v : dY) v = nan;
    return;
  }

  const long double omx2 = std::max(omx2_stable_xy(x), OMx2_MIN);
  const long double y2m1 = std::max(y2m1_stable_xy(y), Y2M1_MIN);

  const double onepnx = 1.0 + nu*x;
  const double A = 1.0 + nu*y;
  const double A_safe = (std::abs(A) < A_MIN) ? (A >= 0.0 ? A_MIN : -A_MIN) : A;

  const long double Gx = omx2 * (long double)onepnx;
  const long double YA = y2m1 * (long double)A_safe; // (y^2-1)A = -G(y) outside

  const long double R2 = (long double)R*(long double)R;

  // invP = D^2/(R^2 Fx)
  const long double invP = D2 / (R2 * (long double)Fx);

  // Derivatives of D^2 (NO 1/D anywhere)
  const long double D2_x = 2.0L*Dld;    // d/dx (D^2) = 2 D
  const long double D2_y = -2.0L*Dld;   // d/dy (D^2) = -2 D

  // invP derivatives (expanded, NO log form)
  const long double Fx_x = (long double)la;
  const long double invP_x = (D2_x / (R2*(long double)Fx)) - (D2 * Fx_x) / (R2*(long double)Fx*(long double)Fx);
  const long double invP_y = (D2_y / (R2*(long double)Fx));

  // -------- contravariant metric components --------
  const long double gxx   = Gx * invP;
  const long double gyy   = YA * invP;
  const long double gphph = D2 / (R2 * Gx);

  // Cc, B
  const double Cc = compute_Cc_cached(la, nu);
  const long double B = 1.0L - (long double)y;

  const long double gtps  = -(long double)Cc * D2 / ((long double)R * (long double)Fx * B * (long double)A_safe);

  // gpsps = Fy * D^2 / (R^2 Fx (y^2-1) A)  = Fy * invP / (y2m1 * A)
  const long double denomYA = y2m1 * (long double)A_safe;
  const long double gpsps = (long double)Fy * invP / denomYA;

  // g^{tt} (Fy-free rational form)
  const long double c = -(long double)y;
  const long double xl = (long double)x;
  const long double lal = (long double)la;
  const long double nul = (long double)nu;

  const long double Npoly =
      -c*c*lal*lal + c*c*lal*nul - c*c*lal + c*c*nul
    + c*lal*lal*nul*xl*xl - 2.0L*c*lal*lal*xl + c*lal*lal
    - c*lal*nul*xl*xl + 4.0L*c*lal*nul*xl - c*lal*nul - 2.0L*c*lal*xl + c*nul - c
    + lal*lal*nul*xl*xl - 2.0L*lal*lal*xl*xl + 2.0L*lal*lal*xl
    + lal*nul*xl*xl - 2.0L*lal*xl + lal - 1.0L;

  const long double Den_tt =
      (c + 1.0L) * (lal - 1.0L) * (c*nul - 1.0L) * (lal*xl + 1.0L);

  if (Den_tt == 0.0L || !std::isfinite((double)Den_tt)) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double &v : dY) v = nan;
    return;
  }

  const long double gtt = Npoly / Den_tt;

  // Derivatives of gtt wrt x and c (then c=-y => dg/dy = -dg/dc)
  const long double Nx =
      2.0L*lal*( c*lal*nul*xl - c*lal - c*nul*xl + 2.0L*c*nul - c
               + lal*nul*xl - 2.0L*lal*xl + lal + nul*xl - 1.0L );

  const long double Nc =
      -2.0L*c*lal*lal + 2.0L*c*lal*nul - 2.0L*c*lal + 2.0L*c*nul
      + lal*lal*nul*xl*xl - 2.0L*lal*lal*xl + lal*lal
      - lal*nul*xl*xl + 4.0L*lal*nul*xl - lal*nul - 2.0L*lal*xl + nul - 1.0L;

  const long double Dx =
      lal * (c + 1.0L) * (lal - 1.0L) * (c*nul - 1.0L);

  const long double Dc =
      (lal - 1.0L) * (lal*xl + 1.0L) * (2.0L*c*nul + nul - 1.0L);

  const long double Den2_tt = Den_tt*Den_tt;

  const long double gtt_x = (Nx*Den_tt - Npoly*Dx) / Den2_tt;
  const long double gtt_c = (Nc*Den_tt - Npoly*Dc) / Den2_tt;

  const long double gtt_dx = gtt_x;
  const long double gtt_dy = -gtt_c;

  // -------- derivatives of other contravariants (ALL expanded) --------

  // Gx_x = d/dx[(1-x^2)(1+nu x)] = (-2x)(1+nu x) + (1-x^2)nu
  const long double Gx_x = (-2.0L*(long double)x) * (long double)onepnx + omx2 * (long double)nu;

  // gxx = Gx*invP
  const long double gxx_x = Gx_x*invP + Gx*invP_x;
  const long double gxx_y = Gx*invP_y;

  // YA = (y^2-1)A
  const long double Y_y = 2.0L*(long double)y;
  const long double A_y = (long double)nu;
  const long double YA_y = Y_y*(long double)A_safe + y2m1*A_y;

  const long double gyy_x = YA*invP_x;
  const long double gyy_y = YA_y*invP + YA*invP_y;

  // gphph = D2/(R2*Gx)
  const long double gphph_x = (D2_x/(R2*Gx)) - (D2*Gx_x)/(R2*Gx*Gx);
  const long double gphph_y = (D2_y/(R2*Gx));

  // gtps = -Cc*D2/(R*Fx*B*A)
  // expanded derivatives (no log form)
  const long double denom_t = (long double)R * (long double)Fx * B * (long double)A_safe;
  const long double gtps_x = -(long double)Cc * ( D2_x/denom_t - (D2*Fx_x)/((long double)R*(long double)Fx*(long double)Fx*B*(long double)A_safe) );
  // y derivative: D2_y and (1/(B*A))_y = +1/(B^2 A) - A_y/(B A^2)
  const long double gtps_y = -(long double)Cc * (
      D2_y/denom_t
    + D2 * ( 1.0L/((long double)R*(long double)Fx*B*B*(long double)A_safe)
           - A_y/((long double)R*(long double)Fx*B*(long double)A_safe*(long double)A_safe) )
  );

  // gpsps = Fy*invP/(y2m1*A)
  // expanded derivatives to avoid Fy'/Fy removable singularity
  const long double Fy_y = (long double)la;
  const long double denom = denomYA;              // y2m1*A
  const long double denom2 = denom*denom;

  const long double denom_y = Y_y*(long double)A_safe + y2m1*A_y; // (y2m1*A)' = Y_y*A + y2m1*A_y
  const long double gpsps_x = (long double)Fy * invP_x / denom;

  const long double termA = (Fy_y*invP + (long double)Fy*invP_y) / denom;
  const long double termB = ((long double)Fy * invP) * (denom_y / denom2);
  const long double gpsps_y = termA - termB;

  // ---------------- dq/ds ----------------
  dY[0] = (double)(gtt*(-(long double)E) + gtps*(long double)Psi);
  dY[1] = (double)(gxx * (long double)px);
  dY[2] = (double)(gyy * (long double)py);
  dY[3] = (double)(gphph * (long double)Phi);
  dY[4] = (double)(gtps*(-(long double)E) + gpsps*(long double)Psi);

  // ---------------- dp/ds ----------------
  dY[5] = (double)(-0.5L*( gtt_dx*(long double)(E*E)
              - 2.0L*gtps_x*(long double)(E*Psi)
              + gpsps_x*(long double)(Psi*Psi)
              + gphph_x*(long double)(Phi*Phi)
              + gxx_x*(long double)(px*px)
              + gyy_x*(long double)(py*py) ));

  dY[6] = (double)(-0.5L*( gtt_dy*(long double)(E*E)
              - 2.0L*gtps_y*(long double)(E*Psi)
              + gpsps_y*(long double)(Psi*Psi)
              + gphph_y*(long double)(Phi*Phi)
              + gxx_y*(long double)(px*px)
              + gyy_y*(long double)(py*py) ));
              
  dY[5] = 0;
              
}

// Event checker: returns
//   0 none,
//   1 horizon,
//   2 escape,
//   3 invalid/out-of-chart
int check_event(const Params& P, const RenderOpts& R, double /*s*/, const Y7& Y)
{
  const double x = Y[1];
  const double y = Y[2];

  if (!std::isfinite(x) || !std::isfinite(y)) return 3;
  if (!(x > -1.0 && x < 1.0)) return 3;
  if (!(y <= -1.0)) return 3;

  const double H = 1.0 + P.nu*y;
  if (H <= R.horizon_eps) return 1;

  const long double D = D_stable_xy(x,y);
  if (!(D > 0.0L)) return 3;

  const long double num = -(long double)y - (long double)x; // (-y) - x
  const long double rho2 = (long double)(P.R*P.R) * num / D;
  if (!std::isfinite((double)rho2) || !(rho2 >= 0.0L)) return 3;

  const long double esc2 = (long double)R.r_escape * (long double)R.r_escape;
  if (rho2 >= esc2) return 2;

  return 0;
}

static inline bool enforce_H0_project_xy(const Params& P, const RayConsts& C, Y7& y)
{
  const double x  = y[1];
  const double yy = y[2];

  if (!std::isfinite(x) || !std::isfinite(yy)) return false;
  // exterior domain: -1 < x < 1, y <= -1
  if (!(x > -1.0 && x < 1.0)) return false;
  if (!(yy <= -1.0)) return false;

  const long double D    = (1.0L + (long double)x) + (-1.0L - (long double)yy);   // x - y
  const long double omx2 = (1.0L - (long double)x) * (1.0L + (long double)x);     // 1-x^2
  const long double y2m1 = ((long double)yy - 1.0L) * ((long double)yy + 1.0L);   // y^2-1

  if (!(D > 0.0L)) return false;

  const long double Fx = 1.0L + (long double)P.lambda*(long double)x;
  const long double Fy = 1.0L + (long double)P.lambda*(long double)yy;
  const long double A  = 1.0L + (long double)P.nu*(long double)yy; // 1+nu y
  const long double B  = 1.0L - (long double)yy;                   // 1 - y
  const long double onepnx = 1.0L + (long double)P.nu*(long double)x;

  if (!std::isfinite((double)Fx) || Fx == 0.0L) return false;

  // safe denominators
  constexpr long double OM_MIN = 1e-300L;
  constexpr long double Y2_MIN = 1e-300L;
  constexpr long double A_MIN  = 1e-15L;

  const long double omx2s = (std::abs(omx2) < OM_MIN) ? (omx2 >= 0 ? OM_MIN : -OM_MIN) : omx2;
  const long double y2m1s = (std::abs(y2m1) < Y2_MIN) ? (y2m1 >= 0 ? Y2_MIN : -Y2_MIN) : y2m1;
  const long double As    = (std::abs(A)   < A_MIN ) ? (A   >= 0 ? A_MIN  : -A_MIN ) : A;

  const long double Gx = omx2s * onepnx;
  const long double YA = y2m1s * As;

  const long double R  = (long double)P.R;
  const long double R2 = R*R;
  const long double D2 = D*D;

  // cache Cc
  thread_local double cache_la = std::numeric_limits<double>::quiet_NaN();
  thread_local double cache_nu = std::numeric_limits<double>::quiet_NaN();
  thread_local double cache_Cc = 0.0;
  if (!(P.lambda == cache_la && P.nu == cache_nu)) {
    cache_la = P.lambda;
    cache_nu = P.nu;
    cache_Cc = std::sqrt(P.lambda*(P.lambda-P.nu) * (1.0+P.lambda)/(1.0-P.lambda));
  }
  const long double Cc = (long double)cache_Cc;

  // invP = D^2/(R^2 Fx)
  const long double invP = D2 / (R2 * Fx);

  // contravariants
  const long double gxx   = Gx * invP;
  const long double gyy   = YA * invP;
  const long double gphph = (Gx != 0.0L) ? (D2 / (R2 * Gx)) : std::numeric_limits<long double>::quiet_NaN();
  const long double gtps  = -Cc * D2 / (R * Fx * B * As);
  const long double gpsps = Fy * D2 / (R2 * Fx * As * y2m1s);

  // g^{tt} (Fy-free rational form using c = -y)
  const long double c   = -(long double)yy;
  const long double xl  = (long double)x;
  const long double lal = (long double)P.lambda;
  const long double nul = (long double)P.nu;

  const long double Npoly =
      -c*c*lal*lal + c*c*lal*nul - c*c*lal + c*c*nul
    + c*lal*lal*nul*xl*xl - 2.0L*c*lal*lal*xl + c*lal*lal
    - c*lal*nul*xl*xl + 4.0L*c*lal*nul*xl - c*lal*nul - 2.0L*c*lal*xl + c*nul - c
    + lal*lal*nul*xl*xl - 2.0L*lal*lal*xl*xl + 2.0L*lal*lal*xl
    + lal*nul*xl*xl - 2.0L*lal*xl + lal - 1.0L;

  const long double Den_tt =
      (c + 1.0L) * (lal - 1.0L) * (c*nul - 1.0L) * (lal*xl + 1.0L);

  if (Den_tt == 0.0L) return false;
  const long double gtt = Npoly / Den_tt;

  // Potential part V(x,y)
  const long double E   = (long double)C.E;   // should be 1
  const long double Phi = (long double)C.Phi;
  const long double Psi = (long double)C.Psi;

  const long double V =
      gtt*(E*E)
    - 2.0L*gtps*(E*Psi)
    + gpsps*(Psi*Psi)
    + gphph*(Phi*Phi);

  // Current momenta
  const long double px = (long double)y[5];
  const long double py = (long double)y[6];

  // Decide which momentum to solve for:
  // near x-poles, gxx is tiny => px is ill-conditioned => solve px from constraint.
  const bool prefer_px = (std::fabs((double)gxx) < std::fabs((double)gyy));

  auto solve_px = [&]() -> bool {
    if (!(gxx > 0.0L)) return false;
    long double rhs = -(V + gyy*(py*py));
    long double px2 = rhs / gxx;

    // tolerate tiny negative from roundoff
    const long double tol = 1e-30L * (1.0L + std::fabs((double)rhs/gxx));
    if (px2 < 0.0L && px2 > -tol) px2 = 0.0L;
    if (!(px2 >= 0.0L) || !std::isfinite((double)px2)) return false;

    const double sgn = (y[5] >= 0.0) ? 1.0 : -1.0;
    y[5] = sgn * std::sqrt((double)px2);
    return std::isfinite(y[5]);
  };

  auto solve_py = [&]() -> bool {
    if (!(gyy > 0.0L)) return false;
    long double rhs = -(V + gxx*(px*px));
    long double py2 = rhs / gyy;

    const long double tol = 1e-30L * (1.0L + std::fabs((double)rhs/gyy));
    if (py2 < 0.0L && py2 > -tol) py2 = 0.0L;
    if (!(py2 >= 0.0L) || !std::isfinite((double)py2)) return false;

    const double sgn = (y[6] >= 0.0) ? 1.0 : -1.0;
    y[6] = sgn * std::sqrt((double)py2);
    return std::isfinite(y[6]);
  };

  bool ok = false;
  if (prefer_px) {
    ok = solve_px();
    if (!ok) ok = solve_py();
  } else {
    ok = solve_py();
    if (!ok) ok = solve_px();
  }
  return ok;
}


// --------------- trace_ray(): apply RESCALE_POW and metric-based phase-space error ---------------
core::RKResult<7> trace_ray(const Params& P, const RenderOpts& R, const core::RKOpts& O,
                           const Y7& y0, const RayConsts& C, double sMax)
{
  auto rhs_base = [&](double s, const Y7& y, Y7& dyds) { rhs(P, C, s, y, dyds); };
  auto ev_fn    = [&](double s, const Y7& y) { return check_event(P, R, s, y); };

  // Project: wrap angles + enforce H=0 by solving for px (algebraic)
  auto proj_fn = [&](Y7& y) {
    project_xy(y);
    if (!enforce_H0_project_xy(P, C, y)) {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      for (double &v : y) v = nan;
      return;
    }
    project_xy(y);
  };

  const int pow = std::clamp(R.rescale_pow, 0, 2);

  // Rescale the entire RHS by f = (x-y)^(-pow)
  auto rhs_fn = [&](double s, const Y7& y, Y7& dyds)
  {
    rhs_base(s, y, dyds);
    for (double v : dyds) if (!std::isfinite(v)) return;

    if (pow == 0) return;

    const long double Dld = D_stable_xy(y[1], y[2]);
    if (!(Dld > 0.0L)) {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      for (double &v : dyds) v = nan;
      return;
    }

    long double f = 1.0L;
    if (pow == 1) f = 1.0L / Dld;
    else          f = 1.0L / (Dld*Dld);

    const long double maxd = (long double)std::numeric_limits<double>::max();
    if (!(f <= maxd)) {
      for (double &v : dyds) v = std::numeric_limits<double>::infinity();
      return;
    }

    const double fd = (double)f;
    for (double &v : dyds) v *= fd;
  };

  // Metric-based error estimator:
  // - q error uses (dx,dy,dphi) with covariant metric
  // - p error uses ONLY p_y (since p_x is algebraic and can diverge)
  // - tighten by 1/(distance to horizon) = 1/max(A, horizon_eps)
  auto err_est = [&](const Y7& /*y*/,
                     const Y7& y_full,
                     const Y7& y_2half,
                     const core::RKOpts& opt) -> double
  {
    constexpr long double OMx2_MIN = 1e-300L;
    constexpr long double Y2M1_MIN = 1e-300L;
    constexpr double A_MIN = 1e-15;

    const double x = y_2half[1];
    const double y = y_2half[2];

    if (!std::isfinite(x) || !std::isfinite(y)) return std::numeric_limits<double>::infinity();
    if (!(x > -1.0 && x < 1.0) || !(y <= -1.0)) return std::numeric_limits<double>::infinity();

    const long double Dld = D_stable_xy(x,y);
    if (!(Dld > 0.0L)) return std::numeric_limits<double>::infinity();
    const long double D2 = Dld*Dld;

    const double la = P.lambda;
    const double nu = P.nu;

    const double Fx = 1.0 + la*x;
    if (!std::isfinite(Fx) || Fx == 0.0) return std::numeric_limits<double>::infinity();

    const long double omx2 = std::max(omx2_stable_xy(x), OMx2_MIN);
    const long double y2m1 = std::max(y2m1_stable_xy(y), Y2M1_MIN);

    const double onepnx = 1.0 + nu*x;
    const double A = 1.0 + nu*y;
    const double A_safe = (std::abs(A) < A_MIN) ? (A >= 0.0 ? A_MIN : -A_MIN) : A;

    const long double Gx = omx2 * (long double)onepnx;
    const long double YA = y2m1 * (long double)A_safe;

    // Covariant metric scalars for dq-norm
    const long double pref = ((long double)P.R*(long double)P.R * (long double)Fx) / D2;
    const long double g_xx_cov   = pref / Gx;
    const long double g_yy_cov   = pref / YA;
    const long double g_phph_cov = ((long double)P.R*(long double)P.R) * Gx / D2;

    const double dx   = y_2half[1] - y_full[1];
    const double dy   = y_2half[2] - y_full[2];
    const double dphi = core::wrap_pi(y_2half[3] - y_full[3]);

    const long double ds2 =
        g_xx_cov*(long double)dx*(long double)dx +
        g_yy_cov*(long double)dy*(long double)dy +
        g_phph_cov*(long double)dphi*(long double)dphi;

    if (!(ds2 >= 0.0L) || !std::isfinite((double)ds2)) return std::numeric_limits<double>::infinity();
    const double ds = std::sqrt((double)ds2);

    // length scale r
    const long double r_sq = ((long double)P.R*(long double)P.R) * (omx2 + y2m1) / D2;
    const double r = std::sqrt(std::max(0.0, (double)r_sq));

    const double sc_len = opt.atol + opt.rtol * std::max(1.0, r);
    const double err_q  = ds / sc_len;

    // Momentum error: ONLY p_y
    const long double invP = D2 / (((long double)P.R*(long double)P.R) * (long double)Fx);
    const long double gyy = YA * invP;

    const double py  = y_2half[6];
    const double dpy = y_2half[6] - y_full[6];

    const long double dp2 = gyy*(long double)dpy*(long double)dpy;
    if (!(dp2 >= 0.0L) || !std::isfinite((double)dp2)) return std::numeric_limits<double>::infinity();

    const double p_mag = std::sqrt(std::max(0.0, (double)(gyy*(long double)py*(long double)py)));
    const double sc_p  = opt.atol + opt.rtol * std::max(1.0, p_mag);
    const double err_p = std::sqrt((double)dp2) / sc_p;

    const double H_safe = std::max(A, R.horizon_eps);
    return std::max(err_q, err_p) / H_safe;
  };

  return core::rk4_adaptive<7>(rhs_fn, ev_fn, err_est, 0.0, y0, sMax, O, proj_fn);
}


core::RKResult<7> trace_ray_dump_csv(const Params& P, const RenderOpts& R, const core::RKOpts& O,
                                     const Y7& y0, const RayConsts& C, double sMax,
                                     const char* csv_path)
{
  std::ofstream out(csv_path ? csv_path : "debug_ray.csv");
  if (!out) throw std::runtime_error("trace_ray_dump_csv: failed to open csv");

  out.setf(std::ios::scientific);
  out << std::setprecision(17);

  // CSV header
  out << "kind,step,s,h,errInf,event,stop_reason,"
         "t,x,y,phi,psi,px,py,"
         "E,Phi,Psi,"
         "dt_ds,dx_ds,dy_ds,dphi_ds,dpsi_ds,dpx_ds,dpy_ds,"
         "H,rescale_f,"
         "D,omx2,y2m1,Fx,Fy,A,B,onepnx,Gx,YA,"
         "gxx,gyy,gphph,gtps,gpsps,gtt\n";

  auto kind_name = [](core::RKObsKind k)->const char* {
    switch (k) {
      case core::RKObsKind::Start:   return "start";
      case core::RKObsKind::BadStep: return "bad";
      case core::RKObsKind::Reject:  return "reject";
      case core::RKObsKind::Accept:  return "accept";
      case core::RKObsKind::Event:   return "event";
      case core::RKObsKind::Stop:    return "stop";
      default: return "unknown";
    }
  };
  auto reason_name = [](core::RKReason r)->const char* {
    switch (r) {
      case core::RKReason::Done:     return "Done";
      case core::RKReason::Event:    return "Event";
      case core::RKReason::Hmin:     return "Hmin";
      case core::RKReason::MaxSteps: return "MaxSteps";
      case core::RKReason::NaN:      return "NaN";
      default: return "Unknown";
    }
  };

  // Same wrappers as trace_ray: rescale entire RHS by (x-y)^(-pow)
  const int pow = std::clamp(R.rescale_pow, 0, 2);

  auto rhs_base = [&](double s, const Y7& y, Y7& dyds) { rhs(P, C, s, y, dyds); };
  auto ev_fn    = [&](double s, const Y7& y) { return check_event(P, R, s, y); };
  //auto proj_fn  = [&](Y7& y) { project_xy(y); };
  auto proj_fn = [&](Y7& y) {
    project_xy(y);                 // wraps angles
    if (!enforce_H0_project_xy(P, C, y)) {
      // poison the state so RK rejects / stops cleanly
      const double nan = std::numeric_limits<double>::quiet_NaN();
      y[0]=nan; y[1]=nan; y[2]=nan; y[3]=nan; y[4]=nan; y[5]=nan; y[6]=nan;
    }
  };


  auto rhs_fn = [&](double s, const Y7& y, Y7& dyds)
  {
    rhs_base(s, y, dyds);
    for (double v : dyds) if (!std::isfinite(v)) return;

    if (pow == 0) return;

    const long double D = (1.0L + (long double)y[1]) + (-1.0L - (long double)y[2]); // stable x-y
    if (!(D > 0.0L)) {
      for (double &v : dyds) v = std::numeric_limits<double>::quiet_NaN();
      return;
    }
    long double f = 1.0L;
    if (pow == 1) f = 1.0L / D;
    else          f = 1.0L / (D*D);

    const double fd = (double)f;
    for (double &v : dyds) v *= fd;
  };

  auto err_est = [&](const Y7& y,
                   const Y7& y_full,
                   const Y7& y_2half,
                   const core::RKOpts& opt) -> double
{
    double errInf = 0.0;
    for (int i=0;i<7;i++) {
      // ignore t, phi, psi and ALSO px (algebraic)
      if (i==0 || i==3 || i==4 || i==5) continue;

      const double err = y_2half[i] - y_full[i];
      const double sc  = opt.atol + opt.rtol * std::max(std::abs(y[i]), std::abs(y_2half[i]));
      errInf = std::max(errInf, std::abs(err)/sc);
    }
    return errInf;
  };

  // Error estimator: keep your existing one here (this is fine for dumping)
  //auto err_est = [&](const Y7& y,
  //                   const Y7& y_full,
  //                   const Y7& y_2half,
  //                   const core::RKOpts& opt) -> double
  //{
  //  return core::default_err_inf_raw<7>(y, y_full, y_2half, opt);
  //};

  // Precompute Cc for metric scalars
  const double la = P.lambda;
  const double nu = P.nu;
  const double Cc = std::sqrt( la*(la-nu) * (1.0+la)/(1.0-la) );

  // Compute contravariant scalars + H robustly (NO derivatives needed)
  auto metric_terms = [&](double x, double y) {
    struct MT {
      bool ok = true;

      long double D=0, omx2=0, y2m1=0, Fx=0, Fy=0, A=0, B=0, onepnx=0, Gx=0, YA=0;
      long double gxx=0, gyy=0, gphph=0, gtps=0, gpsps=0, gtt=0;
      long double rescale_f=1;
    } mt;

    // stable basics
    mt.D    = (1.0L + (long double)x) + (-1.0L - (long double)y);
    mt.omx2 = (1.0L - (long double)x) * (1.0L + (long double)x);
    mt.y2m1 = ((long double)y - 1.0L) * ((long double)y + 1.0L);

    mt.Fx = 1.0L + (long double)la*(long double)x;
    mt.Fy = 1.0L + (long double)la*(long double)y;
    mt.onepnx = 1.0L + (long double)nu*(long double)x;
    mt.A = 1.0L + (long double)nu*(long double)y;
    mt.B = 1.0L - (long double)y;

    // rescale factor used by integrator
    if (pow == 0) mt.rescale_f = 1.0L;
    else if (pow == 1) mt.rescale_f = (mt.D != 0.0L) ? (1.0L/mt.D) : std::numeric_limits<long double>::infinity();
    else mt.rescale_f = (mt.D != 0.0L) ? (1.0L/(mt.D*mt.D)) : std::numeric_limits<long double>::infinity();

    // guardrails
    if (!(mt.D > 0.0L) || !std::isfinite((double)mt.D)) mt.ok = false;

    // avoid 0/0 in these scalars: use tiny floors ONLY for denominators
    constexpr long double OM_MIN  = 1e-300L;
    constexpr long double Y2_MIN  = 1e-300L;
    constexpr long double A_MIN   = 1e-15L;

    const long double omx2s = (std::abs(mt.omx2) < OM_MIN) ? ( (mt.omx2>=0)? OM_MIN : -OM_MIN ) : mt.omx2;
    const long double y2m1s = (std::abs(mt.y2m1) < Y2_MIN) ? ( (mt.y2m1>=0)? Y2_MIN : -Y2_MIN ) : mt.y2m1;
    const long double As    = (std::abs(mt.A)   < A_MIN ) ? ( (mt.A  >=0)? A_MIN  : -A_MIN  ) : mt.A;

    mt.Gx = omx2s * mt.onepnx;
    mt.YA = y2m1s * As;

    const long double R  = (long double)P.R;
    const long double R2 = R*R;
    const long double D2 = mt.D*mt.D;

    // invP = D^2/(R^2 Fx)
    if (mt.Fx == 0.0L || !std::isfinite((double)mt.Fx)) mt.ok = false;
    const long double invP = (mt.Fx != 0.0L) ? (D2 / (R2 * mt.Fx)) : std::numeric_limits<long double>::quiet_NaN();

    // g^{xx}, g^{yy}, g^{phi phi}
    mt.gxx = mt.Gx * invP;
    mt.gyy = mt.YA * invP;
    mt.gphph = (mt.Gx != 0.0L) ? (D2 / (R2 * mt.Gx)) : std::numeric_limits<long double>::quiet_NaN();

    // gtps, gpsps
    mt.gtps  = -(long double)Cc * D2 / (R * mt.Fx * mt.B * As);
    mt.gpsps = mt.Fy * D2 / (R2 * mt.Fx * As * y2m1s);

    // g^{tt} (Fy-free rational form using c = -y)
    const long double c  = -(long double)y;
    const long double xl = (long double)x;
    const long double lal = (long double)la;
    const long double nul = (long double)nu;

    const long double Npoly =
        -c*c*lal*lal + c*c*lal*nul - c*c*lal + c*c*nul
      + c*lal*lal*nul*xl*xl - 2.0L*c*lal*lal*xl + c*lal*lal
      - c*lal*nul*xl*xl + 4.0L*c*lal*nul*xl - c*lal*nul - 2.0L*c*lal*xl + c*nul - c
      + lal*lal*nul*xl*xl - 2.0L*lal*lal*xl*xl + 2.0L*lal*lal*xl
      + lal*nul*xl*xl - 2.0L*lal*xl + lal - 1.0L;

    const long double Den_tt =
        (c + 1.0L) * (lal - 1.0L) * (c*nul - 1.0L) * (lal*xl + 1.0L);

    mt.gtt = (Den_tt != 0.0L) ? (Npoly / Den_tt) : std::numeric_limits<long double>::quiet_NaN();

    return mt;
  };

  // Observer: write one line per accept/reject/bad/stop
  auto obs = [&](const core::RKObsInfo<7>& info)
  {
    const Y7& y = *(info.y);

    const double t   = y[0];
    const double x   = y[1];
    const double yy  = y[2];
    const double phi = y[3];
    const double psi = y[4];
    const double px  = y[5];
    const double py  = y[6];

    // metric scalars + rescale factor
    auto mt = metric_terms(x, yy);

    // derivatives at this state (what the integrator is using)
    Y7 dyds{};
    rhs_fn(info.s, y, dyds);

    // Hamiltonian constraint H (PHYSICAL, unscaled)
    // p_t = -E
    const long double E   = (long double)C.E;
    const long double Phi = (long double)C.Phi;
    const long double Psi = (long double)C.Psi;

    const long double pxl = (long double)px;
    const long double pyl = (long double)py;

    long double H = std::numeric_limits<long double>::quiet_NaN();
    if (mt.ok && std::isfinite((double)mt.gtt) && std::isfinite((double)mt.gtps) &&
        std::isfinite((double)mt.gpsps) && std::isfinite((double)mt.gphph) &&
        std::isfinite((double)mt.gxx) && std::isfinite((double)mt.gyy))
    {
      const long double term =
          mt.gtt*(E*E)
        - 2.0L*mt.gtps*(E*Psi)
        + mt.gpsps*(Psi*Psi)
        + mt.gphph*(Phi*Phi)
        + mt.gxx*(pxl*pxl)
        + mt.gyy*(pyl*pyl);
      H = 0.5L*term;
    }

    const char* stop_reason = (info.kind == core::RKObsKind::Stop) ? reason_name(info.stop_reason) : "";

    out << kind_name(info.kind) << ","
        << info.step << ","
        << info.s << ","
        << info.h << ","
        << info.errInf << ","
        << info.event_id << ","
        << stop_reason << ","
        << t << "," << x << "," << yy << "," << phi << "," << psi << "," << px << "," << py << ","
        << (double)E << "," << (double)Phi << "," << (double)Psi << ","
        << dyds[0] << "," << dyds[1] << "," << dyds[2] << "," << dyds[3] << "," << dyds[4] << "," << dyds[5] << "," << dyds[6] << ","
        << (double)H << "," << (double)mt.rescale_f << ","
        << (double)mt.D << "," << (double)mt.omx2 << "," << (double)mt.y2m1 << ","
        << (double)mt.Fx << "," << (double)mt.Fy << "," << (double)mt.A << "," << (double)mt.B << ","
        << (double)mt.onepnx << "," << (double)mt.Gx << "," << (double)mt.YA << ","
        << (double)mt.gxx << "," << (double)mt.gyy << "," << (double)mt.gphph << ","
        << (double)mt.gtps << "," << (double)mt.gpsps << "," << (double)mt.gtt
        << "\n";
  };

  return core::rk4_adaptive_observe<7>(rhs_fn, ev_fn, err_est, 0.0, y0, sMax, O, proj_fn, obs);
}


} // namespace singly_rotating_black_ring













