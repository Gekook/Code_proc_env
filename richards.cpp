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
 * A simple linear retention `theta = a * psi` is used for demonstration and
 * clipped to physical bounds.
 */
double theta(double psi) {
    double th = a * psi;
    if (th < 0.0) th = 0.0;
    if (th > theta_sat) th = theta_sat;
    return th;
}
/**
 * @brief Derivative of the retention curve with respect to pressure head.
 * @param psi Pressure head [m].
 * @return dtheta/dpsi (zero outside the linear region due to clipping).
 */
double dtheta_dpsi(double psi) {
    double th = a * psi;
    if (th < 0.0 || th > theta_sat) return 0.0;
    return a;
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
    // Variable PSI profile: sinusoidal gradient + offset
    std::vector<double> psi(Nz);
    for (int i = 0; i < Nz; ++i)
        psi[i] = 0.5 + 0.5 * std::sin(M_PI * i * dz); // can change as desired

    // Constant K profile (modify here for a stratified domain)
    std::vector<double> k_profile(Nz, 1e-2);

    const int max_newton_iter = 30;
    const double tol = 1e-8;

    for (int n = 0; n < Nt; ++n) {
        Eigen::VectorXd psi_new = Eigen::VectorXd::Map(psi.data(), Nz);

        for (int iter = 0; iter < max_newton_iter; ++iter) {
            Eigen::VectorXd S = Eigen::VectorXd::Zero(Nz);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(Nz, Nz);

            // Dirichlet boundary conditions
            S(0) = psi_new(0) - psi[0];
            S(Nz-1) = psi_new(Nz-1) - psi[Nz-1];
            J(0,0) = 1.0;
            J(Nz-1,Nz-1) = 1.0;

            // Interior nodes
            for (int i = 1; i < Nz-1; ++i) {
                double th_i = theta(psi_new(i));
                double th_ip1 = theta(psi_new(i+1));
                double th_im1 = theta(psi_new(i-1));
                double dth_i = dtheta_dpsi(psi_new(i));
                double dth_ip1 = dtheta_dpsi(psi_new(i+1));
                double dth_im1 = dtheta_dpsi(psi_new(i-1));

                // Half-points - geometric mean on the provided k values
                double K_iphalf = std::sqrt(k_profile[i] * k_profile[i + 1]);
                double K_iminushalf = std::sqrt(k_profile[i] * k_profile[i - 1]);
                double theta_iphalf = 0.5 * (th_i + th_ip1);
                double theta_iminushalf = 0.5 * (th_i + th_im1);

                // Fluxes
                double F_iphalf_p = theta_iphalf * K_iphalf * ((psi_new(i+1) - psi_new(i)) / dz);
                double F_iminushalf_p = theta_iminushalf * K_iminushalf * ((psi_new(i) - psi_new(i-1)) / dz);
                double F_iphalf_g = theta_iphalf * K_iphalf * g;
                double F_iminushalf_g = theta_iminushalf * K_iminushalf * g;
                double F_iphalf = F_iphalf_p + F_iphalf_g;
                double F_iminushalf = F_iminushalf_p + F_iminushalf_g;

                // Newton residual
                S(i) = th_i - theta(psi[i]) - (dt/dz)*(F_iphalf - F_iminushalf);

                // Jacobian: ONLY derivatives of THETA (dK/dpsi = 0 because k is fixed)
                // == [PSI_(i-1)]
                if (i-1 >= 0) {
                    double dtheta_iminushalf_dpsi_im1 = 0.5 * dth_im1;
                    double der = - (dt/dz) * (
                          dtheta_iminushalf_dpsi_im1 * K_iminushalf * ((psi_new(i) - psi_new(i-1)) / dz)
                        - theta_iminushalf * K_iminushalf / (2.0*dz)
                        + dtheta_iminushalf_dpsi_im1 * K_iminushalf * g
                    );
                    J(i, i-1) = der;
                }
                // == [PSI_(i+1)]
                if (i+1 < Nz) {
                    double dtheta_iphalf_dpsi_ip1 = 0.5 * dth_ip1;
                    double der = (dt/dz) * (
                          dtheta_iphalf_dpsi_ip1 * K_iphalf * ((psi_new(i+1) - psi_new(i)) / dz)
                        + theta_iphalf * K_iphalf / (2.0*dz)
                        + dtheta_iphalf_dpsi_ip1 * K_iphalf * g
                    );
                    J(i, i+1) = der;
                }
                // == [PSI_(i)]
                double dtheta_iphalf_dpsi_i = 0.5 * dth_i;
                double dtheta_iminushalf_dpsi_i = 0.5 * dth_i;
                double der_c = dth_i;
                der_c -= (dt/dz) * (
                      - dtheta_iphalf_dpsi_i * K_iphalf * ((psi_new(i+1) - psi_new(i)) / dz)
                      - theta_iphalf * K_iphalf / (2.0*dz)
                      + dtheta_iphalf_dpsi_i * K_iphalf * g
                      + dtheta_iminushalf_dpsi_i * K_iminushalf * ((psi_new(i) - psi_new(i-1)) / dz)
                      + theta_iminushalf * K_iminushalf / (2.0*dz)
                      + dtheta_iminushalf_dpsi_i * K_iminushalf * g
                );
                J(i, i) = der_c;
            }

            // SOLUTION: use Eigen QR (colPivHouseholderQr)
            Eigen::VectorXd delta_psi = J.colPivHouseholderQr().solve(S);
            double max_corr = delta_psi.cwiseAbs().maxCoeff();
            psi_new -= delta_psi;
            if (max_corr < tol) break;
        }

        for (int i = 0; i < Nz; ++i)
            psi[i] = psi_new(i);

// Replace the output in main with:
if (n % 50 == 0) {
    std::ofstream csvout("snapshot_" + std::to_string(n) + ".csv");
    csvout << "z,psi,theta\n";
    for (int i = 0; i < Nz; ++i) {
        csvout << i*dz << "," << psi[i] << "," << theta(psi[i]) << "\n";
    }
    csvout.close();
}

    }

    return 0;
}

