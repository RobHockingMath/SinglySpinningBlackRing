#include "singly_rotating_black_ring.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <array>

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

// ------------------------------
// Stable helpers for new chart:
//   x = tanh(chi)
//   y = -cosh(eta)
//   D = x - y = tanh(chi) + cosh(eta) = (1+tanh) + (cosh-1)
// ------------------------------
static inline double tanh_plus_one_stable(double chi) {
  // 1 + tanh(chi) = 2/(1+e^{-2chi}) but avoid overflow.
  if (chi >= 0.0) {
    const double e = std::exp(-2.0*chi); // in (0,1]
    return 2.0 / (1.0 + e);
  } else {
    const double e = std::exp( 2.0*chi); // in (0,1]
    return 2.0 * e / (1.0 + e);
  }
}

static inline double sech2_stable(double chi) {
  // sech^2(chi) = 4 t / (1+t)^2 with t = exp(-2|chi|)
  const double a = std::abs(chi);
  const double t = std::exp(-2.0*a);   // [0,1]
  const double d = 1.0 + t;
  return 4.0 * t / (d*d);
}

static inline long double coshm1_stable(double eta) {
  // cosh(eta)-1 = 2*sinh^2(eta/2) (stable near eta=0)
  const long double h = 0.5L*(long double)eta;
  const long double sh = std::sinh(h);
  return 2.0L*sh*sh;
}

static inline long double D_stable(double chi, double eta) {
  // D = tanh(chi) + cosh(eta) = (1+tanh) + (cosh-1)
  return (long double)tanh_plus_one_stable(chi) + coshm1_stable(eta);
}

struct RingPointXY {
  double x, y, phi, psi;
  double r1, r2, Sigma, A;
};

// --- original cart4 -> ring (x,y,phi,psi) map (kept; not a metric thing) ---
static RingPointXY cart4_to_ring_xy(const Vec4& p4, double R) {
  // CONVENTION:
  //   psi = atan2(x2,x1) (x1x2 plane), r2 = sqrt(x1^2+x2^2)
  //   phi = atan2(x4,x3) (x3x4 plane), r1 = sqrt(x3^2+x4^2)
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

  // d(x,y)/d(r1,r2) for Sigma = sqrt((r1^2+r2^2+R^2)^2 - 4 R^2 r2^2)
  const double dx_dr1 = -4*R*R*r1*(r1*r1 - r2*r2 + R*R)/Sig3;
  const double dx_dr2 = -8*R*R*(r1*r1)*r2              /Sig3;

  const double dy_dr1 =  8*R*R*r1*(r2*r2)              /Sig3;
  const double dy_dr2 =  4*R*R*r2*(r2*r2 - r1*r1 - R*R)/Sig3;

  // dr1/dxi : r1 = sqrt(x3^2+x4^2)
  Vec4 dr1{}; dr1[0]=0; dr1[1]=0; dr1[2]=x3/r1; dr1[3]=x4/r1;
  // dr2/dxi : r2 = sqrt(x1^2+x2^2)
  Vec4 dr2{}; dr2[0]=x1/r2; dr2[1]=x2/r2; dr2[2]=0; dr2[3]=0;

  // dphi/dxi : phi = atan2(x4,x3)
  Vec4 dphi{}; dphi[0]=0; dphi[1]=0; dphi[2]= -x4/(r1*r1); dphi[3]= x3/(r1*r1);
  // dpsi/dxi : psi = atan2(x2,x1)
  Vec4 dpsi{}; dpsi[0]= -x2/(r2*r2); dpsi[1]= x1/(r2*r2); dpsi[2]=0; dpsi[3]=0;

  Vec4 dx{}, dy{};
  for (int i=0;i<4;i++) {
    dx[i] = dx_dr1*dr1[i] + dx_dr2*dr2[i];
    dy[i] = dy_dr1*dr1[i] + dy_dr2*dr2[i];
  }

  Mat4 J = Mat4::zero();
  // rows: x, y, phi, psi
  for (int c=0;c<4;c++) { J(0,c)=dx[c]; J(1,c)=dy[c]; J(2,c)=dphi[c]; J(3,c)=dpsi[c]; }
  return J;
}

// --- covariant metric in (t,chi,eta,phi,psi) (used for frame build only) ---
static Mat5 metric_srb_chieta_cov(double chi, double eta, double R, double lambda, double nu)
{
  const double x = std::tanh(chi);
  const double omx2 = sech2_stable(chi); // 1-x^2

  const long double U_ld = coshm1_stable(eta);
  const double cH = 1.0 + (double)U_ld;
  const double sH = std::sinh(eta);
  const double sH2 = sH*sH;

  const double y  = -cH;

  const double Fx = 1.0 + lambda*x;
  const double Fy = 1.0 + lambda*y;

  const double A = 1.0 + nu*y;   // = 1 - nu cosh

  const long double D = D_stable(chi, eta);
  if (!(D > 0.0L)) throw std::runtime_error("D<=0 in metric");

  const double Cc = std::sqrt(lambda*(lambda-nu) * (1.0+lambda)/(1.0-lambda));

  const long double D2 = D*D;
  const long double pref_ld = ((long double)R*(long double)R * (long double)Fx) / D2;
  const double pref = (double)pref_ld;

  const double onepy = 1.0 + y; // = 1 - cosh = -U
  const double Aterm = (Cc*R*onepy) / Fy;

  Mat5 g = Mat5::zero();

  // indices: 0=t,1=chi,2=eta,3=phi,4=psi
  g(0,0) = -Fy/Fx;

  g(0,4) = (Fy/Fx) * Aterm;
  g(4,0) = g(0,4);

  const double minusGy_over_Fy = (sH2 * A) / Fy;

  g(4,4)  = -(Fy/Fx) * (Aterm*Aterm);
  g(4,4) += pref * minusGy_over_Fy;

  const double onepnx = 1.0 + nu*x;
  g(1,1) = pref * (omx2 / onepnx);

  g(2,2) = pref * (1.0 / A);

  g(3,3) = pref * (omx2 * onepnx / Fx);

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
    const double uu = inner_g(u,u,g); // ~ -1
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

  // Get (x,y,phi,psi) from Euclidean point, then convert to (chi,eta)
  RingPointXY rp = cart4_to_ring_xy(p4, P.R);

  // chi = atanh(x), eta = acosh(-y)
  double x = rp.x;
  constexpr double xeps = 1e-15;
  x = std::clamp(x, -1.0 + xeps, 1.0 - xeps);
  const double chi = std::atanh(x);

  const double ym = std::max(1.0, -rp.y);
  const double eta = std::acosh(ym);

  // Jacobian factors:
  // dchi/dx = 1/(1-x^2)
  // deta/dy = -1/sqrt(y^2-1) = -1/sinh(eta)
  const double omx2 = 1.0 - x*x;
  if (omx2 < 1e-14) throw std::runtime_error("1-x^2 ~ 0 (chi chart ill-conditioned at camera)");
  const double dchi_dx = 1.0 / omx2;

  const double sinh_eta = std::sinh(eta);
  if (sinh_eta < 1e-14) throw std::runtime_error("sinh(eta) ~ 0 (eta chart ill-conditioned at camera)");
  const double deta_dy = -1.0 / sinh_eta;

  Vec5 q{};
  q[0]=0.0;
  q[1]=chi;
  q[2]=eta;
  q[3]=rp.phi;
  q[4]=rp.psi;

  Mat4 Jxy = cart4_to_ring_xy_jacobian(p4, P.R, rp.r1, rp.r2, rp.Sigma);

  Mat4 J = Mat4::zero();
  for (int c=0;c<4;c++) {
    J(0,c) = dchi_dx  * Jxy(0,c);
    J(1,c) = deta_dy  * Jxy(1,c);
    J(2,c) = Jxy(2,c);
    J(3,c) = Jxy(3,c);
  }

  std::array<Vec5,4> V{};
  for (int k=0;k<4;k++) {
    Vec4 ek = core::col(E4, k);
    Vec4 qv = core::mul(J, ek); // [dchi,deta,dphi,dpsi]
    Vec5 vk{};
    vk[0]=0.0;
    vk[1]=qv[0];
    vk[2]=qv[1];
    vk[3]=qv[2];
    vk[4]=qv[3];
    V[k]=vk;
  }

  Mat5 g = metric_srb_chieta_cov(chi, eta, P.R, P.lambda, P.nu);
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

  // force future-directed
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
  y0[1] = F.q[1];
  y0[2] = F.q[2];
  y0[3] = F.q[3];
  y0[4] = F.q[4];
  y0[5] = p[1]; // p_chi
  y0[6] = p[2]; // p_eta
  return true;
}

// =====================================================
// RHS in (chi,eta), Hamiltonian first-order form
// State: [t, chi, eta, phi, psi, p_chi, p_eta]
// =====================================================
void rhs(const Params& P, const RayConsts& C, double /*s*/, const Y7& Y, Y7& dY)
{
  constexpr double OMx2_MIN = 1e-300;
  constexpr double SH2_MIN  = 1e-300;
  constexpr double A_MIN    = 1e-15;

  const double chi = Y[1];
  const double eta = Y[2];

  const double pchi = Y[5];
  const double peta = Y[6];

  const double R  = P.R;
  const double la = P.lambda;
  const double nu = P.nu;

  const double E   = C.E;   // should be 1
  const double Phi = C.Phi;
  const double Psi = C.Psi;

  const double x = std::tanh(chi);
  const double omx2 = sech2_stable(chi);                 // sech^2(chi)
  const double omx2_safe = std::max(omx2, OMx2_MIN);

  const long double U_ld = coshm1_stable(eta);           // cosh-1
  const long double cH_ld = 1.0L + U_ld;                 // cosh
  const double cH = (double)cH_ld;
  const double sH = std::sinh(eta);                      // sinh
  const double sH2 = sH*sH;
  const double sH2_safe = std::max(sH2, SH2_MIN);

  const double y = -cH;

  const double Fx = 1.0 + la*x;
  const double Fy = 1.0 + la*y;                          // = 1 - la*cosh

  const double B = cH + 1.0;                             // 1+cosh
  const double A = 1.0 + nu*y;                           // = 1 - nu*cosh
  const double A_safe = (std::abs(A) < A_MIN) ? (A >= 0.0 ? A_MIN : -A_MIN) : A;

  // cache Cc
  thread_local double cache_la = std::numeric_limits<double>::quiet_NaN();
  thread_local double cache_nu = std::numeric_limits<double>::quiet_NaN();
  thread_local double cache_Cc = 0.0;
  if (!(la == cache_la && nu == cache_nu)) {
    cache_la = la;
    cache_nu = nu;
    cache_Cc = std::sqrt( la*(la-nu) * (1.0+la)/(1.0-la) );
  }
  const double Cc  = cache_Cc;

  const long double D  = D_stable(chi, eta);
  if (!(D > 0.0L) || !std::isfinite((double)D) || !std::isfinite(Fx) || Fx == 0.0) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double &v : dY) v = nan;
    return;
  }
  const long double D2 = D*D;
  const double invD = (double)(1.0L / D);

  const double R2 = R*R;

  // invP = D^2/(R^2 Fx)
  const double invP = (double)(D2 / ((long double)R2 * (long double)Fx));

  const double onepnx = 1.0 + nu*x;

  // -------- contravariant metric components (stable) --------
  const double g_chichi = (onepnx / omx2_safe) * invP;
  const double g_etaeta = A * invP;
  const double g_phph   = (double)(D2 / ((long double)R2 * (long double)omx2_safe * (long double)onepnx));

  const double gtps   = -(Cc * (double)D2) / (R * Fx * B * A_safe);
  const double gpsps  = (Fy * (double)D2) / (R2 * Fx * sH2_safe * A_safe);

  // -------- g^{tt} and its derivatives WITHOUT Fy in denom --------
  // Use simplified rational form in (x,c=cosh eta):
  //   gtt = N(x,c) / ((c+1)*(la-1)*(c*nu-1)*(la*x+1))
  // where N is a polynomial (exactly cancels any Fy→0 issues).
  const long double c = cH_ld;
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

  if (!std::isfinite((double)Den_tt) || Den_tt == 0.0L) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double &v : dY) v = nan;
    return;
  }

  const double gtt = (double)(Npoly / Den_tt);

  // Derivatives wrt x and c:
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

  // Chain rule:
  // dx/dchi = sech^2(chi) = omx2
  // dc/deta = sinh(eta)   = sH
  const double gtt_chi = (double)(gtt_x * (long double)omx2);
  const double gtt_eta = (double)(gtt_c * (long double)sH);

  // -------- derivatives for other contravariants (avoid 0*inf) --------
  const double Fx_chi = la * omx2;
  const double D_chi  = omx2;
  const double D_eta  = sH;

  // invP derivatives
  const double dln_invP_chi = 2.0 * D_chi * invD - (Fx_chi / Fx);
  const double dln_invP_eta = 2.0 * D_eta * invD;

  const double invP_chi = invP * dln_invP_chi;
  const double invP_eta = invP * dln_invP_eta;

  // g_chichi = ratio * invP, ratio = onepnx/omx2_safe
  const double onepnx_chi = nu * omx2;
  const double domx2_chi  = -2.0 * x * omx2;

  double domx2_safe_chi = 0.0;
  if (omx2 > OMx2_MIN * 10.0) domx2_safe_chi = domx2_chi;

  const double ratio = onepnx / omx2_safe;
  const double ratio_chi = (onepnx_chi*omx2_safe - onepnx*domx2_safe_chi) / (omx2_safe*omx2_safe);

  const double gchichi_chi = ratio_chi * invP + ratio * invP_chi;
  const double gchichi_eta = ratio * invP_eta;

  // g_etaeta = A * invP
  const double A_eta = -nu * sH;
  const double getaeta_chi = A * invP_chi;
  const double getaeta_eta = A_eta * invP + A * invP_eta;

  // g_phph = D^2/(R^2 omx2_safe onepnx)
  const double gphph_chi = g_phph * ( 2.0*D_chi*invD
                                     - ( (omx2 > OMx2_MIN*10.0) ? (domx2_chi/omx2_safe) : 0.0 )
                                     - (onepnx_chi/onepnx) );

  const double gphph_eta = g_phph * (2.0*D_eta*invD);

  // gtps = -Cc D^2/(R Fx B A_safe)
  const double B_eta = sH;
  const double dln_gtps_chi = 2.0*D_chi*invD - (Fx_chi/Fx);
  const double dln_gtps_eta = 2.0*D_eta*invD - (B_eta/B) - (A_eta/A_safe);

  const double gtps_chi = gtps * dln_gtps_chi;
  const double gtps_eta = gtps * dln_gtps_eta;

  // gpsps = Fy * invP / (sH2_safe * A_safe)
  const double Fy_eta = -la * sH;

  const double gpsps_chi = (Fy / (sH2_safe * A_safe)) * invP_chi;

  double sH2_eta = 0.0;
  if (sH2 > SH2_MIN * 10.0) sH2_eta = 2.0*sH*cH;

  const double term1 = (Fy_eta*invP + Fy*invP_eta) / (sH2_safe * A_safe);
  const double term2 = (sH2_eta != 0.0) ? (Fy*invP * (sH2_eta) / (sH2_safe*sH2_safe * A_safe)) : 0.0;
  const double term3 = Fy*invP * (A_eta) / (sH2_safe * A_safe*A_safe);

  const double gpsps_eta = term1 - term2 - term3;

  // ---------------- dq/ds ----------------
  dY[0] = gtt*(-E) + gtps*Psi;
  dY[1] = g_chichi * pchi;
  dY[2] = g_etaeta * peta;
  dY[3] = g_phph   * Phi;
  dY[4] = gtps*(-E) + gpsps*Psi;

  // ---------------- dp/ds ----------------
  dY[5] = -0.5*( gtt_chi*(E*E)
              - 2.0*gtps_chi*(E*Psi)
              + gpsps_chi*(Psi*Psi)
              + gphph_chi*(Phi*Phi)
              + gchichi_chi*(pchi*pchi)
              + getaeta_chi*(peta*peta) );

  dY[6] = -0.5*( gtt_eta*(E*E)
              - 2.0*gtps_eta*(E*Psi)
              + gpsps_eta*(Psi*Psi)
              + gphph_eta*(Phi*Phi)
              + gchichi_eta*(pchi*pchi)
              + getaeta_eta*(peta*peta) );
}

// Event checker: returns
//   0 none,
//   1 horizon,
//   2 escape,
//   3 invalid/out-of-chart (will show as magenta)
int check_event(const Params& P, const RenderOpts& R, double /*s*/, const Y7& Y)
{
  const double chi = Y[1];
  const double eta = Y[2];

  if (!std::isfinite(chi) || !std::isfinite(eta)) return 3;
  if (eta < 0.0) return 3;

  const double x = std::tanh(chi);

  const long double U_ld = coshm1_stable(eta);
  const double cH = 1.0 + (double)U_ld;
  const double y  = -cH;

  const double H = 1.0 + P.nu*y; // = 1 - nu*cosh
  if (H <= R.horizon_eps) return 1;

  const long double D = D_stable(chi, eta);
  if (!(D > 0.0L)) return 3;

  const long double num = (long double)cH - (long double)x;
  const long double rho2 = (long double)(P.R*P.R) * num / D;
  if (!std::isfinite((double)rho2) || !(rho2 >= 0.0L)) return 3;

  const long double esc2 = (long double)R.r_escape * (long double)R.r_escape;
  if (rho2 >= esc2) return 2;

  return 0;
}

// (same as your current singly_rotating_black_ring.cpp from the last replacement I gave you)
// ...everything above unchanged...

// --------------- ONLY CHANGED PART IS INSIDE trace_ray() ---------------
core::RKResult<7> trace_ray(const Params& P, const RenderOpts& R, const core::RKOpts& O,
                           const Y7& y0, const RayConsts& C, double sMax)
{
  auto rhs_fn = [&](double s, const Y7& y, Y7& dyds) { rhs(P, C, s, y, dyds); };
  auto ev_fn  = [&](double s, const Y7& y) { return check_event(P, R, s, y); };

  auto proj_fn = [&](Y7& y) { fold_chieta(y); };

  // ---------- METRIC-BASED ERROR ESTIMATOR + 1/dist_to_horizon ----------
  auto err_est = [&](const Y7& /*y*/,
                     const Y7& y_full,
                     const Y7& y_2half,
                     const core::RKOpts& opt) -> double
  {
    auto domain_ok = [&](const Y7& yy)->bool {
      const double chi = yy[1];
      const double eta = yy[2];
      if (!std::isfinite(chi) || !std::isfinite(eta)) return false;
      if (eta < 0.0) return false;

      const long double U_ld = coshm1_stable(eta);
      const double cH = 1.0 + (double)U_ld;
      const double yv = -cH;
      const double H = 1.0 + P.nu*yv;   // dist-to-horizon factor (A)
      if (!(H > 0.0)) return false;

      const long double D = D_stable(chi, eta);
      if (!(D > 0.0L)) return false;

      return true;
    };

    if (!domain_ok(y_full) || !domain_ok(y_2half)) {
      return std::numeric_limits<double>::infinity();
    }

    // Evaluate covariant diagonal (chi,eta,phi) metric scalars at y_2half.
    constexpr double OMx2_MIN = 1e-300;
    constexpr double A_MIN    = 1e-15;

    const double chi = y_2half[1];
    const double eta = y_2half[2];

    const double x = std::tanh(chi);
    const double omx2 = sech2_stable(chi);
    const double omx2_safe = std::max(omx2, OMx2_MIN);

    const long double U_ld = coshm1_stable(eta);
    const long double cH_ld = 1.0L + U_ld;
    const double cH = (double)cH_ld;
    const double sH = std::sinh(eta);

    const double yv = -cH;

    const double Fx = 1.0 + P.lambda*x;
    const double onepnx = 1.0 + P.nu*x;

    const double H = 1.0 + P.nu*yv; // = 1 - nu*cosh(eta)
    const double H_safe = std::max(H, R.horizon_eps);

    const double A = H;
    const double A_safe = (std::abs(A) < A_MIN) ? (A >= 0.0 ? A_MIN : -A_MIN) : A;

    const long double D = D_stable(chi, eta);
    const long double D2 = D*D;

    const long double pref = ((long double)P.R*(long double)P.R * (long double)Fx) / D2;

    const long double g_chichi_cov = pref * (long double)omx2_safe / (long double)onepnx;
    const long double g_etaeta_cov = pref * 1.0L / (long double)A_safe;
    const long double g_phph_cov   = ((long double)P.R*(long double)P.R) * (long double)omx2_safe * (long double)onepnx / D2;

    const double dchi = y_2half[1] - y_full[1];
    const double deta = y_2half[2] - y_full[2];
    const double dphi = core::wrap_pi(y_2half[3] - y_full[3]);

    const long double ds2 =
        g_chichi_cov*(long double)dchi*(long double)dchi +
        g_etaeta_cov*(long double)deta*(long double)deta +
        g_phph_cov  *(long double)dphi*(long double)dphi;

    if (!(ds2 >= 0.0L) || !std::isfinite((double)ds2)) {
      return std::numeric_limits<double>::infinity();
    }
    const double ds = std::sqrt((double)ds2);

    // Physical length scale: r = sqrt(r1^2+r2^2), with r1 = R*sech/D, r2 = R*sinh/D
    const long double invD = 1.0L / D;
    const long double sech = std::sqrt((long double)omx2_safe);
    const long double r1 = (long double)P.R * sech * invD;
    const long double r2 = (long double)P.R * (long double)sH * invD;
    const double r = (double)std::sqrt(r1*r1 + r2*r2);

    // Base scale
    const double sc = opt.atol + opt.rtol * std::max(1.0, r);

    // Horizon-weight: multiply error by 1/dist_to_horizon
    // (equivalently divide the tolerance by dist_to_horizon)
    return (ds / sc) / H_safe;
  };

  return core::rk4_adaptive<7>(rhs_fn, ev_fn, err_est, 0.0, y0, sMax, O, proj_fn);
}


} // namespace singly_rotating_black_ring











