/**************************************************************************
 * @file richards.cpp
 * @brief 1D Richards-equation toy solver using Eigen for linear algebra.
 *
 * Toy finite-volume discretization + implicit Newton solver for a
 * nonlinear Richards-like equation in 1D.
 *
 * Setup:
 *  - Initial condition: uniform theta ≈ 0.32 in the whole domain
 *    (so psi is uniform too).
 *  - Top boundary: Dirichlet "wet" psi = psi_top (acqua che entra dall'alto).
 *  - Bottom boundary: zero physical flux F = theta K (dpsi/dz + g) = 0.
 *
 * Flux (with gravity):
 *  F = theta(psi) * K * (dpsi/dz + g)
 *
 *************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

// Spatial and temporal parameters
const double dz = 0.1;
const double dt = 5.0;
const int    Nz = 63;
const int    Nt = 10000;

// Gravity (used in the flux)
const double g = 9.81;

// Physical parameters (toy)
const double theta_sat = 0.50;   // saturation water content

// Hydraulic conductivity profile (constant toy value)
const double K_const = 1e-2;     // un po' più grande per vedere evoluzione

// Constitutive relations
// theta(psi) = (psi/c)^(1/m) clipped in [0, theta_sat], psi > 0
double theta(double psi, int m = 2, double c = 1.0)
{
    if (psi <= 0.0) return 0.0;

    double th = std::pow(psi / c, 1.0 / m);
    if (th >= theta_sat) return theta_sat;
    return th;
}

// d theta / d psi
double dtheta_dpsi(double psi, int m = 2, double c = 1.0)
{
    if (psi <= 0.0) return 0.0;

    double th = std::pow(psi / c, 1.0 / m);
    if (th >= theta_sat) return 0.0;  // derivata zero nella zona saturata (clipping)

    double dth = (1.0 / m) * (1.0 / c) * std::pow(psi / c, 1.0 / m - 1.0);
    return dth;
}

int main()
{
    // -------------------------------
    // Inizializzazione stato iniziale
    // -------------------------------
    // Vogliamo theta_init ≈ 0.32 ovunque
    const double theta_init = 0.32;
    const double psi_init   = theta_init * theta_init;   // inversa di theta = sqrt(psi)

    // Top molto bagnato (saturazione): qualsiasi psi_top > 0 porta theta = theta_sat
    const double psi_top = 1.0;

    std::cout << std::setprecision(8);
    std::cout << "theta_init = " << theta_init << "\n";
    std::cout << "psi_init   = " << psi_init   << "\n";
    std::cout << "psi_top    = " << psi_top    << "\n";

    // Campo psi iniziale: uniforme
    std::vector<double> psi(Nz, psi_init);
    for(double i=0; i<Nz; ++i){
        psi[i]=psi_init*(i/Nz);
    }

    // K costante
    std::vector<double> k_profile(Nz, K_const);

    const int    max_newton_iter = 30;
    const double tol             = 1e-8;

    // -------------------------------
    // Time stepping
    // -------------------------------
    for (int n = 0; n < Nt; ++n) {

        // psi_new = incognita al tempo n+1, psi = psi^n fissato
        Eigen::VectorXd psi_new = Eigen::VectorXd::Map(psi.data(), Nz);

        // ---------------------------
        // Newton-Raphson iteration
        // ---------------------------
        for (int iter = 0; iter < max_newton_iter; ++iter) {

            Eigen::VectorXd S = Eigen::VectorXd::Zero(Nz);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(Nz, Nz);

            // ==================================
            // BOUNDARY CONDITIONS
            // ==================================

            // TOP: Dirichlet bagnato psi = psi_top
            S(0)   = psi_new(0) - psi_top;
            J(0,0) = 1.0;

            // BOTTOM: zero physical flux F = theta K (dpsi/dz + g) = 0
            // Half-cell tra gli ultimi due nodi: i = Nz-2 e Nz-1
            {
                int i_b   = Nz - 1;     // nodo di fondo
                int i_in  = Nz - 2;     // nodo interno subito sopra

                double th_b   = theta(psi_new(i_b));
                double th_in  = theta(psi_new(i_in));
                double dth_b  = dtheta_dpsi(psi_new(i_b));
                double dth_in = dtheta_dpsi(psi_new(i_in));

                double theta_bhalf = 0.5 * (th_b + th_in);
                double K_bhalf     = std::sqrt(k_profile[i_b] * k_profile[i_in]);

                double dpsi_b = (psi_new(i_b) - psi_new(i_in)) / dz;

                // Flusso al fondo: F_bottom = theta_bhalf * K_bhalf * (dpsi_b + g)
                double F_bottom = theta_bhalf * K_bhalf * (dpsi_b + g);

                // Residuo: imponiamo F_bottom = 0
                S(i_b) = F_bottom;

                // Derivate di F_bottom per il Jacobiano
                // dF/dpsi_b
                //   = K_bhalf [ 0.5 * dth_b * (dpsi_b + g) + theta_bhalf * (1/dz) ]
                double dF_dpsi_b =
                    K_bhalf * (0.5 * dth_b * (dpsi_b + g) + theta_bhalf / dz);

                // dF/dpsi_in
                //   = K_bhalf [ 0.5 * dth_in * (dpsi_b + g) - theta_bhalf * (1/dz) ]
                double dF_dpsi_in =
                    K_bhalf * (0.5 * dth_in * (dpsi_b + g) - theta_bhalf / dz);

                J(i_b, i_b)  = dF_dpsi_b;
                J(i_b, i_in) = dF_dpsi_in;
            }

            // ==================================
            // INTERIOR NODES: 1 .. Nz-2
            // ==================================
            for (int i = 1; i < Nz - 1; ++i) {

                double th_i   = theta(psi_new(i));
                double th_ip1 = theta(psi_new(i + 1));
                double th_im1 = theta(psi_new(i - 1));

                double dth_i   = dtheta_dpsi(psi_new(i));
                double dth_ip1 = dtheta_dpsi(psi_new(i + 1));
                double dth_im1 = dtheta_dpsi(psi_new(i - 1));

                // Hydraulic conductivity on half-cells (geometric mean)
                double K_iphalf      = std::sqrt(k_profile[i] * k_profile[i + 1]);
                double K_iminushalf  = std::sqrt(k_profile[i] * k_profile[i - 1]);

                // theta on half-cells (simple average)
                double theta_iphalf      = 0.5 * (th_i + th_ip1);
                double theta_iminushalf  = 0.5 * (th_i + th_im1);

                // Gradiente di psi
                double dpsi_p = (psi_new(i + 1) - psi_new(i)) / dz;    // forward
                double dpsi_m = (psi_new(i)     - psi_new(i - 1)) / dz; // backward

                // ======================================
                // FLUX WITH GRAVITY:
                // F = theta * K * (dpsi/dz + g)
                // ======================================
                double F_iphalf     = theta_iphalf     * K_iphalf     * (dpsi_p + g);
                double F_iminushalf = theta_iminushalf * K_iminushalf * (dpsi_m + g);

                // Newton residual:
                // theta(psi^{n+1}) - theta(psi^n) - (dt/dz)(F_{i+1/2}^{n+1} - F_{i-1/2}^{n+1}) = 0
                S(i) = th_i - theta(psi[i]) - (dt / dz) * (F_iphalf - F_iminushalf);

                // ==================================
                // Jacobian entries
                // ==================================

                // F_{i-1/2} = theta_iminushalf * K_iminushalf * (dpsi_m + g)
                // dpsi_m / dpsi_{i-1} = -1 / dz
                // d theta_iminushalf / dpsi_{i-1} = 0.5 * dth_im1
                double dF_iminushalf_dpsi_im1 =
                      0.5 * dth_im1 * K_iminushalf * (dpsi_m + g)
                    - theta_iminushalf * K_iminushalf / dz;

                // F_{i+1/2} = theta_iphalf * K_iphalf * (dpsi_p + g)
                // dpsi_p / dpsi_{i+1} =  1 / dz
                // d theta_iphalf / dpsi_{i+1} = 0.5 * dth_ip1
                double dF_iphalf_dpsi_ip1 =
                      0.5 * dth_ip1 * K_iphalf * (dpsi_p + g)
                    + theta_iphalf * K_iphalf / dz;

                // dF_{i+1/2} / dpsi_i
                // dpsi_p / dpsi_i = -1 / dz
                // d theta_iphalf / dpsi_i = 0.5 * dth_i
                double dF_iphalf_dpsi_i =
                      0.5 * dth_i * K_iphalf * (dpsi_p + g)
                    - theta_iphalf * K_iphalf / dz;

                // dF_{i-1/2} / dpsi_i
                // dpsi_m / dpsi_i =  1 / dz
                // d theta_iminushalf / dpsi_i = 0.5 * dth_i
                double dF_iminushalf_dpsi_i =
                      0.5 * dth_i * K_iminushalf * (dpsi_m + g)
                    + theta_iminushalf * K_iminushalf / dz;

                // Assemble Jacobian
                // J(i, i-1)
                J(i, i - 1) = (dt / dz) * dF_iminushalf_dpsi_im1;

                // J(i, i+1)
                J(i, i + 1) = - (dt / dz) * dF_iphalf_dpsi_ip1;

                // J(i, i)
                double der_c = dth_i
                    - (dt / dz) * (dF_iphalf_dpsi_i - dF_iminushalf_dpsi_i);
                J(i, i) = der_c;
            }

            // RISOLVO il sistema lineare: J * delta_psi = S
            Eigen::VectorXd delta_psi = J.colPivHouseholderQr().solve(S);
            double max_corr = delta_psi.cwiseAbs().maxCoeff();

            psi_new -= delta_psi;

            if (max_corr < tol) {
                // Newton converged
                break;
            }
        }

        // Update in time: psi^n <- psi^{n+1}
        for (int i = 0; i < Nz; ++i) {
            psi[i] = psi_new(i);
        }

        // Output periodico su file + log di controllo
        if (n % 100 == 0) {
            // stampa qualche info
            double theta_max = 0.0;
            double theta_bottom_now = theta(psi[Nz - 1]);
            for (int i = 0; i < Nz; ++i) {
                double th_i = theta(psi[i]);
                if (th_i > theta_max) theta_max = th_i;
            }
            std::cout << "t = " << n * dt
                      << "  theta_bottom = " << theta_bottom_now
                      << "  theta_max = " << theta_max << "\n";

            // salva profilo
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
