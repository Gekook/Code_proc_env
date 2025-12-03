/**************************************************************************
 * @file richards.cpp
 * @brief 1D Richards-equation toy solver using Eigen for linear algebra.
 *
 * This file contains a simple finite-volume discretization and an implicit
 * Newton solver for the nonlinear update at each time step. The
 * implementation is intentionally minimal and uses a fixed hydraulic
 * conductivity profile for clarity.
 *************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

// Spatial and temporal parameters
const double dz = 0.1;
const double dt = 0.01;
const int Nz = 63;
const int Nt = 1000;
const double g = 9.81;

// Physical parameters
const double a = 0.08;
const double theta_sat = 0.1;

// Constitutive relations
/**
 * @brief Volumetric water content (retention curve).
 * @param psi Pressure head [m].
 * @return Volumetric water content (dimensionless), clipped to [0, theta_sat].
 *
 * Toy retention curve: theta = (psi/c)^(1/m), clipped in [0, theta_sat].
 */
double theta(double psi, int m=2, double c=1.0) {
    if (psi <= 0.0) return 0.0;

    double th = std::pow(psi/c, 1.0/m);
    if (th >= theta_sat) return theta_sat;

    return th;
}

double dtheta_dpsi(double psi, int m=2, double c=1.0) {
    if (psi <= 0.0) return 0.0;

    double th = std::pow(psi/c, 1.0/m);
    if (th >= theta_sat) return 0.0;  // derivata zero nella zona clip

    double dth = (1.0/m) * (1.0/c) * std::pow(psi/c, 1.0/m - 1.0);
    return dth;
}


/**
 * @brief Main program: set up fields and run time-stepping with Newton solve.
 *
 * Initializes a spatial psi profile and a constant hydraulic conductivity
 * profile. At each time step a Newton-Raphson iteration assembles the
 * residual vector `S` and Jacobian matrix `J` and solves the linear system
 * using Eigen's solver. Snapshots are written periodically to CSV files.
 */
int main() {
    // Initial PSI profile
    std::vector<double> psi(Nz, 0.0);
    for (int i = 0; i < 4; ++i)
        psi[i] = 0.0001;   // piccolo psi positivo ⇒ theta < theta_sat

    // Constant K profile (modify here for a stratified domain)
    std::vector<double> k_profile(Nz, 1e-2);

    const int max_newton_iter = 30;
    const double tol = 1e-8;

    for (int n = 0; n < Nt; ++n) {

        // psi_new = incognita al tempo n+1, psi = psi^n fissato
        Eigen::VectorXd psi_new = Eigen::VectorXd::Map(psi.data(), Nz);

        for (int iter = 0; iter < max_newton_iter; ++iter) {
            Eigen::VectorXd S = Eigen::VectorXd::Zero(Nz);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(Nz, Nz);

            // Dirichlet boundary conditions (psi_new = psi_old ai bordi)
            S(0)      = psi_new(0)      - psi[0];
            S(Nz - 1) = psi_new(Nz - 1) - psi[Nz - 1];
            J(0, 0)           = 1.0;
            J(Nz - 1, Nz - 1) = 1.0;

            // Interior nodes
            for (int i = 1; i < Nz - 1; ++i) {
                double th_i   = theta(psi_new(i));
                double th_ip1 = theta(psi_new(i + 1));
                double th_im1 = theta(psi_new(i - 1));

                double dth_i   = dtheta_dpsi(psi_new(i));
                double dth_ip1 = dtheta_dpsi(psi_new(i + 1));
                double dth_im1 = dtheta_dpsi(psi_new(i - 1));

                // Half-points - geometric mean on the provided k values
                double K_iphalf      = std::sqrt(k_profile[i] * k_profile[i + 1]);
                double K_iminushalf  = std::sqrt(k_profile[i] * k_profile[i - 1]);
                double theta_iphalf  = 0.5 * (th_i + th_ip1);
                double theta_iminushalf = 0.5 * (th_i + th_im1);

                // Gradiente di psi
                double dpsi_p = (psi_new(i + 1) - psi_new(i)) / dz;
                double dpsi_m = (psi_new(i)     - psi_new(i - 1)) / dz;

                // Fluxes
                double F_iphalf_p     = theta_iphalf     * K_iphalf     * dpsi_p;
                double F_iminushalf_p = theta_iminushalf * K_iminushalf * dpsi_m;
                double F_iphalf_g     = th_ip1 * K_iphalf     * g;
                double F_iminushalf_g = th_i   * K_iminushalf * g;
                double F_iphalf       = F_iphalf_p     + F_iphalf_g;
                double F_iminushalf   = F_iminushalf_p + F_iminushalf_g;

                // Newton residual: θ(ψ^{n+1}) - θ(ψ^n) - (dt/dz)(F_{i+1/2}^{n+1} - F_{i-1/2}^{n+1})
                S(i) = th_i - theta(psi[i]) - (dt/dz) * (F_iphalf - F_iminushalf);

                // =========================
                // Jacobian entries (COERENTI col residuo)
                // =========================

                // ∂F_{i-1/2}/∂ψ_{i-1}
                double dF_iminushalf_dpsi_im1 =
                      0.5 * dth_im1 * K_iminushalf * dpsi_m
                    - theta_iminushalf * K_iminushalf / dz;
                // J(i,i-1) = (dt/dz) * ∂F_{i-1/2}/∂ψ_{i-1}
                J(i, i - 1) = (dt/dz) * dF_iminushalf_dpsi_im1;

                // ∂F_{i+1/2}/∂ψ_{i+1}
                double dF_iphalf_dpsi_ip1 =
                      0.5 * dth_ip1 * K_iphalf * dpsi_p
                    + theta_iphalf * K_iphalf / dz
                    + dth_ip1 * K_iphalf * g;  // da F_iphalf_g
                // J(i,i+1) = - (dt/dz) * ∂F_{i+1/2}/∂ψ_{i+1}
                J(i, i + 1) = - (dt/dz) * dF_iphalf_dpsi_ip1;

                // ∂F_{i+1/2}/∂ψ_i
                double dF_iphalf_dpsi_i =
                      0.5 * dth_i * K_iphalf * dpsi_p
                    - theta_iphalf * K_iphalf / dz;
                // ∂F_{i-1/2}/∂ψ_i
                double dF_iminushalf_dpsi_i =
                      0.5 * dth_i * K_iminushalf * dpsi_m
                    + theta_iminushalf * K_iminushalf / dz
                    + dth_i * K_iminushalf * g;  // da F_iminushalf_g

                // J(i,i) = dθ_i/dψ_i - (dt/dz) * (∂F_{i+1/2}/∂ψ_i - ∂F_{i-1/2}/∂ψ_i)
                double der_c = dth_i
                    - (dt/dz) * (dF_iphalf_dpsi_i - dF_iminushalf_dpsi_i);
                J(i, i) = der_c;
            }

            // SOLUZIONE del sistema lineare: J * delta_psi = S
            Eigen::VectorXd delta_psi = J.colPivHouseholderQr().solve(S);
            double max_corr = delta_psi.cwiseAbs().maxCoeff();

            psi_new -= delta_psi;

            if (max_corr < tol) break;
        }

        // Update in time: psi^n ← psi^{n+1}
        for (int i = 0; i < Nz; ++i)
            psi[i] = psi_new(i);

        // Output periodico
        if (n % 50 == 0) {
            std::ofstream csvout("snapshot_" + std::to_string(n) + ".csv");
            csvout << "z,psi,theta\n";
            for (int i = 0; i < Nz; ++i) {
                csvout << i * dz << "," << psi[i] << "," << theta(psi[i]) << "\n";
            }
            csvout.close();
        }
    }

    return 0;
}