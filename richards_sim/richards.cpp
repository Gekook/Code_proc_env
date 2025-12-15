/**************************************************************************
 * @file richards_python_constants_with_neumann_bc.cpp
 * @brief 1D Richards-equation toy solver using Eigen for linear algebra,
 *        with constants aligned to the provided Python code and Neumann BC
 *        at the bottom (dpsi/dz = 0, psi_{Nz-1} = psi_{Nz-2}).
 *************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

// -------------------------------
// Constants aligned to Python
// -------------------------------
static constexpr double C_const = 1.0;     // C = 1.0
static constexpr int    m_const = 2;       // m = 2
static constexpr double g_const = 9.81;    // g = 9.81

static constexpr double L = 1.0;          // domain length
static constexpr int    Nz = 201;         // Nz = 201
static constexpr double dt = 5.0;         // dt = 5.0
static constexpr double t_final = 1000.0; // t_final = 1000.0

static constexpr double dz = L / (Nz - 1); // dz = L/(Nz-1)
static constexpr int Nt = (int)std::ceil(t_final / dt); // Nt = ceil(t_final/dt) = 200

static constexpr double theta_sat = 0.35; // Theta_max in Python
static constexpr double K_const = 2e-4;   // K0 in Python

// Python IC parameters
static constexpr double theta_init_top = 0.05;
static constexpr double theta_init_bottom = 0.30;

// Python top BC parameters
static constexpr double theta_top_imposed = 1.05; // (as in Python, even if > theta_sat)
static constexpr double psi_top = C_const * theta_top_imposed * theta_top_imposed; // C*theta^m

// -------------------------------
// Constitutive relations (same shape as Python)
// theta(psi) = min( (max(psi,0)/C)^(1/m), theta_sat )
// dtheta/dpsi = 0 if psi<=0 or saturated
// -------------------------------
double theta(double psi)
{
    if (psi <= 0.0) return 0.0;

    double th = std::pow(psi / C_const, 1.0 / m_const);
    return (th >= theta_sat) ? theta_sat : th;
}

double dtheta_dpsi(double psi)
{
    if (psi <= 0.0) return 0.0;

    double th = std::pow(psi / C_const, 1.0 / m_const);
    if (th >= theta_sat) return 0.0; // clipped region

    // d/dpsi [(psi/C)^(1/m)] = (1/m) * (1/C)^(1/m) * psi^(1/m - 1)
    // With C=1, this reduces to (1/m)*psi^(1/m-1), but keep general form:
    double dth = (1.0 / m_const) * std::pow(1.0 / C_const, 1.0 / m_const)
                 * std::pow(psi, 1.0 / m_const - 1.0);
    return dth;
}

int main()
{
    std::cout << std::setprecision(10);

    std::cout << "Aligned constants (Python):\n";
    std::cout << "  L=" << L << " Nz=" << Nz << " dz=" << dz << "\n";
    std::cout << "  dt=" << dt << " t_final=" << t_final << " Nt=" << Nt << "\n";
    std::cout << "  C=" << C_const << " m=" << m_const << " g=" << g_const << "\n";
    std::cout << "  K=" << K_const << " theta_sat=" << theta_sat << "\n";
    std::cout << "  theta_top_imposed=" << theta_top_imposed << " -> psi_top=" << psi_top << "\n\n";

    // -------------------------------
    // Initial condition aligned to Python:
    // theta0(z) linear from theta_init_top (z=0) to theta_init_bottom (z=L)
    // psi0 = C * theta^m (unsat inversion)
    // -------------------------------
    std::vector<double> psi(Nz, 0.0);
    for (int k = 0; k < Nz; ++k) {
        double zfrac = (double)k / (double)(Nz - 1);
        double th0 = theta_init_top * (1.0 - zfrac) + theta_init_bottom * zfrac;

        // invert unsat law: psi = C * theta^m
        double psi0 = C_const * std::pow(th0, (double)m_const);

        // if th0 >= theta_sat, clamp psi to saturation threshold (same idea as Python)
        if (th0 >= theta_sat) {
            psi0 = C_const * std::pow(theta_sat, (double)m_const);
        }

        psi[k] = psi0;
    }

    // K profile constant (aligned to Python)
    std::vector<double> k_profile(Nz, K_const);

    const int    max_newton_iter = 30;
    const double tol = 1e-8;

    // -------------------------------
    // Time stepping
    // -------------------------------
    for (int n = 0; n < Nt; ++n) {

        Eigen::VectorXd psi_new = Eigen::VectorXd::Map(psi.data(), Nz);

        // Newton iterations
        for (int iter = 0; iter < max_newton_iter; ++iter) {

            Eigen::VectorXd S = Eigen::VectorXd::Zero(Nz);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(Nz, Nz);

            // ---------------------------
            // TOP boundary: Dirichlet psi = psi_top
            // ---------------------------
            S(0)   = psi_new(0) - psi_top;
            J(0,0) = 1.0;

            // ---------------------------
            // BOTTOM boundary (Neumann: dpsi/dz = 0, psi_{Nz-1} = psi_{Nz-2})
            // ---------------------------
            {
                int i_b  = Nz - 1;
                int i_in = Nz - 2;

                // Neumann BC (no gradient at bottom)
                S(i_b)      = psi_new(i_b) - psi_new(i_in);
                J(i_b,i_b)  = 1.0;
                J(i_b,i_in) = -1.0;
            }

            // ---------------------------
            // Interior nodes: 1 .. Nz-2
            // ---------------------------
            for (int i = 1; i < Nz - 1; ++i) {

                double th_i   = theta(psi_new(i));
                double th_ip1 = theta(psi_new(i + 1));
                double th_im1 = theta(psi_new(i - 1));

                double dth_i   = dtheta_dpsi(psi_new(i));
                double dth_ip1 = dtheta_dpsi(psi_new(i + 1));
                double dth_im1 = dtheta_dpsi(psi_new(i - 1));

                double K_iphalf     = std::sqrt(k_profile[i] * k_profile[i + 1]);
                double K_iminushalf = std::sqrt(k_profile[i] * k_profile[i - 1]);

                double theta_iphalf     = 0.5 * (th_i + th_ip1);
                double theta_iminushalf = 0.5 * (th_i + th_im1);

                double dpsi_p = (psi_new(i + 1) - psi_new(i)) / dz;
                double dpsi_m = (psi_new(i)     - psi_new(i - 1)) / dz;

                double F_iphalf     = theta_iphalf     * K_iphalf     * (dpsi_p + g_const);
                double F_iminushalf = theta_iminushalf * K_iminushalf * (dpsi_m + g_const);

                // residual: theta(psi^{n+1}) - theta(psi^n) - (dt/dz)(F_{i+1/2}-F_{i-1/2}) = 0
                S(i) = th_i - theta(psi[i]) - (dt / dz) * (F_iphalf - F_iminushalf);

                // Jacobian terms
                double dF_iminushalf_dpsi_im1 =
                      0.5 * dth_im1 * K_iminushalf * (dpsi_m + g_const)
                    - theta_iminushalf * K_iminushalf / dz;

                double dF_iphalf_dpsi_ip1 =
                      0.5 * dth_ip1 * K_iphalf * (dpsi_p + g_const)
                    + theta_iphalf * K_iphalf / dz;

                double dF_iphalf_dpsi_i =
                      0.5 * dth_i * K_iphalf * (dpsi_p + g_const)
                    - theta_iphalf * K_iphalf / dz;

                double dF_iminushalf_dpsi_i =
                      0.5 * dth_i * K_iminushalf * (dpsi_m + g_const)
                    + theta_iminushalf * K_iminushalf / dz;

                J(i, i - 1) = (dt / dz) * dF_iminushalf_dpsi_im1;
                J(i, i + 1) = - (dt / dz) * dF_iphalf_dpsi_ip1;

                J(i, i) = dth_i - (dt / dz) * (dF_iphalf_dpsi_i - dF_iminushalf_dpsi_i);
            }

            // Solve linear system
            Eigen::VectorXd delta_psi = J.colPivHouseholderQr().solve(S);
            double max_corr = delta_psi.cwiseAbs().maxCoeff();

            psi_new -= delta_psi;

            if (max_corr < tol) break;
        }

        // Update psi
        for (int i = 0; i < Nz; ++i) psi[i] = psi_new(i);

        // Output (every 10 steps is enough now, since Nt=200)
        if (n % 10 == 0) {
            double theta_max_now = 0.0;
            double theta_min_now = 1e100;
            for (int i = 0; i < Nz; ++i) {
                double th = theta(psi[i]);
                theta_max_now = std::max(theta_max_now, th);
                theta_min_now = std::min(theta_min_now, th);
            }

            std::cout << "t=" << (n * dt)
                      << "  theta_min=" << theta_min_now
                      << "  theta_max=" << theta_max_now
                      << "  psi_min=" << *std::min_element(psi.begin(), psi.end())
                      << "  psi_max=" << *std::max_element(psi.begin(), psi.end())
                      << "\n";

            std::ofstream csvout("snapshot_" + std::to_string(n) + ".csv");
            csvout << "z,psi,theta\n";
            for (int i = 0; i < Nz; ++i) {
                double z = i * dz;
                csvout << z << "," << psi[i] << "," << theta(psi[i]) << "\n";
            }
            csvout.close();
        }
    }

    return 0;
}
