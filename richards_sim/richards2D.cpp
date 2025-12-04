/**************************************************************************
 * @file richards_2d.cpp
 * @brief 2D Richards-equation toy solver using Eigen for linear algebra.
 *
 * Estende il solver 1D in uno spazio (x,z). Discretizzazione a volumi finiti
 * e schema implicito in tempo con Newton. I flussi sono considerati in x e z;
 * in direzione z è presente la gravità, in x no.
 *
 * I risultati sono salvati in una serie di CSV con colonne:
 *    x, z, t, psi, theta
 * così da poter plottare mappe di intensità e animazioni temporali.
 *************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

// Spatial and temporal parameters
const double dx = 0.1;
const double dz = 0.1;
const int Nx   = 25;
const int Nz   = 25;
const double dt = 1;
const double g  = 9.81;
const int Nt   = 100;

// Physical parameters
const double a = 0.08;
const double theta_sat = 0.1;

// Convenience: total number of nodes
const int Ntot = Nx * Nz;

// Indexing helper: (ix,iz) -> linear index
inline int idx(int ix, int iz) {
    return iz + Nz * ix; // col-major in (z), then x
}

// Constitutive relations
/**
 * @brief Volumetric water content (retention curve).
 * @param psi Pressure head [m].
 * @return Volumetric water content (dimensionless), clipped to [0, theta_sat].
 *
 * Toy retention curve: theta = (psi/c)^(1/m), clipped in [0, theta_sat].
 */
double theta(double psi, int m = 2, double c = 1.0) {
    if (psi <= 0.0) return 0.0;

    double th = std::pow(psi / c, 1.0 / m);
    if (th >= theta_sat) return theta_sat;

    return th;
}

double dtheta_dpsi(double psi, int m = 2, double c = 1.0) {
    if (psi <= 0.0) return 0.0;

    double th = std::pow(psi / c, 1.0 / m);
    if (th >= theta_sat) return 0.0;  // derivata zero nella zona clip

    double dth = (1.0 / m) * (1.0 / c) * std::pow(psi / c, 1.0 / m - 1.0);
    return dth;
}

/**
 * @brief Main program: 2D fields and time-stepping with Newton solve.
 *
 * Inizializza un profilo 2D psi(x,z) e un profilo di conducibilità idraulica
 * costante. Ad ogni passo di tempo esegue una iterazione di Newton
 * costruendo residuo S e jacobiano J (Ntot x Ntot) e risolvendo il sistema
 * lineare con Eigen. Scrive snapshot in CSV periodicamente.
 */
int main() {

    // Initial PSI profile: piccolo psi positivo in una fascia vicino al fondo
    std::vector<double> psi(Ntot, 0.0);
    for (int ix = 0; ix < Nx; ++ix) {
            psi[idx(ix, 0)] = 0.0001;  
    }

    // Constant K profile (modify here for a stratified 2D domain)
    std::vector<double> k_profile(Ntot, 1e-2);

    const int max_newton_iter = 10;
    const double tol = 1e-8;

    for (int n = 0; n < Nt; ++n) {

        // psi_new = incognita al tempo n+1, psi = psi^n fissato
        Eigen::VectorXd psi_new = Eigen::VectorXd::Map(psi.data(), Ntot);

        for (int iter = 0; iter < max_newton_iter; ++iter) {
            Eigen::VectorXd S = Eigen::VectorXd::Zero(Ntot);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(Ntot, Ntot);

            // Loop su tutti i nodi
            for (int ix = 0; ix < Nx; ++ix) {
                for (int iz = 0; iz < Nz; ++iz) {
                    int k = idx(ix, iz);

                    // --- Boundary: Dirichlet psi_new = psi_old ---
                    if (ix == 0 || ix == Nx - 1 || iz == 0 || iz == Nz - 1) {
                        S(k)      = psi_new(k) - psi[k];
                        J(k, k)   = 1.0;
                        continue;
                    }

                    // --- Interior node ---
                    int kL = idx(ix - 1, iz);
                    int kR = idx(ix + 1, iz);
                    int kD = idx(ix, iz - 1);
                    int kU = idx(ix, iz + 1);

                    double psi_i  = psi_new(k);
                    double psi_L  = psi_new(kL);
                    double psi_R  = psi_new(kR);
                    double psi_D  = psi_new(kD);
                    double psi_U  = psi_new(kU);

                    double th_i   = theta(psi_i);
                    double th_L   = theta(psi_L);
                    double th_R   = theta(psi_R);
                    double th_D   = theta(psi_D);
                    double th_U   = theta(psi_U);

                    double dth_i  = dtheta_dpsi(psi_i);
                    double dth_L  = dtheta_dpsi(psi_L);
                    double dth_R  = dtheta_dpsi(psi_R);
                    double dth_D  = dtheta_dpsi(psi_D);
                    double dth_U  = dtheta_dpsi(psi_U);

                    double K_i  = k_profile[k];
                    double K_L  = k_profile[kL];
                    double K_R  = k_profile[kR];
                    double K_D  = k_profile[kD];
                    double K_U  = k_profile[kU];

                    // --- Flussi in direzione x (senza gravità) ---

                    // Faccia destra (ix+1/2)
                    double K_xp = std::sqrt(K_i * K_R);
                    double theta_xp = 0.5 * (th_i + th_R);
                    double dpsi_xp  = (psi_R - psi_i) / dx;
                    double F_xp     = theta_xp * K_xp * dpsi_xp;

                    // Faccia sinistra (ix-1/2)
                    double K_xm = std::sqrt(K_i * K_L);
                    double theta_xm = 0.5 * (th_i + th_L);
                    double dpsi_xm  = (psi_i - psi_L) / dx;
                    double F_xm     = theta_xm * K_xm * dpsi_xm;

                    // --- Flussi in direzione z (con gravità) ---

                    // Faccia superiore (iz+1/2)
                    double K_zp = std::sqrt(K_i * K_U);
                    double theta_zp = 0.5 * (th_i + th_U);
                    double dpsi_zp  = (psi_U - psi_i) / dz;
                    double F_zp_p   = theta_zp * K_zp * dpsi_zp;
                    double F_zp_g   = th_U * K_zp * g;
                    double F_zp     = F_zp_p + F_zp_g;

                    // Faccia inferiore (iz-1/2)
                    double K_zm = std::sqrt(K_i * K_D);
                    double theta_zm = 0.5 * (th_i + th_D);
                    double dpsi_zm  = (psi_i - psi_D) / dz;
                    double F_zm_p   = theta_zm * K_zm * dpsi_zm;
                    double F_zm_g   = th_i * K_zm * g;
                    double F_zm     = F_zm_p + F_zm_g;

                    // --- Residuo di Newton in (ix,iz) ---
                    // S = θ(ψ^{n+1}) - θ(ψ^n)
                    //     - dt * [ (F_x+ - F_x-) / dx + (F_z+ - F_z-) / dz ]
                    double theta_old = theta(psi[k]);  // psi^n
                    S(k) = th_i - theta_old
                         - dt * ( (F_xp - F_xm) / dx + (F_zp - F_zm) / dz );

                    // =========================
                    // Jacobian entries (coerenti col residuo)
                    // =========================

                    // --- Derivate dei flussi in x ---

                    // ∂F_xm/∂psi_L (ix-1/2)
                    double dF_xm_dpsi_L =
                          0.5 * dth_L * K_xm * dpsi_xm
                        - theta_xm * K_xm / dx;
                    // ∂F_xm/∂psi_i
                    double dF_xm_dpsi_i =
                          0.5 * dth_i * K_xm * dpsi_xm
                        + theta_xm * K_xm / dx;

                    // ∂F_xp/∂psi_R (ix+1/2)
                    double dF_xp_dpsi_R =
                          0.5 * dth_R * K_xp * dpsi_xp
                        + theta_xp * K_xp / dx;
                    // ∂F_xp/∂psi_i
                    double dF_xp_dpsi_i =
                          0.5 * dth_i * K_xp * dpsi_xp
                        - theta_xp * K_xp / dx;

                    // --- Derivate dei flussi in z ---

                    // ∂F_zm/∂psi_D (iz-1/2)
                    double dF_zm_dpsi_D =
                          0.5 * dth_D * K_zm * dpsi_zm
                        - theta_zm * K_zm / dz;
                    // ∂F_zm/∂psi_i
                    double dF_zm_dpsi_i =
                          0.5 * dth_i * K_zm * dpsi_zm
                        + theta_zm * K_zm / dz
                        + dth_i * K_zm * g; // da F_zm_g

                    // ∂F_zp/∂psi_U (iz+1/2)
                    double dF_zp_dpsi_U =
                          0.5 * dth_U * K_zp * dpsi_zp
                        + theta_zp * K_zp / dz
                        + dth_U * K_zp * g; // da F_zp_g
                    // ∂F_zp/∂psi_i
                    double dF_zp_dpsi_i =
                          0.5 * dth_i * K_zp * dpsi_zp
                        - theta_zp * K_zp / dz;

                    // --- Assemblaggio Jacobiano ---

                    // Diagonale J(k,k)
                    double der_c = dth_i
                        - dt * ( (dF_xp_dpsi_i - dF_xm_dpsi_i) / dx
                               + (dF_zp_dpsi_i - dF_zm_dpsi_i) / dz );
                    J(k, k) = der_c;

                    // Vicini in x
                    J(k, kL) =  dt * dF_xm_dpsi_L / dx;   // contrib. da F_xm
                    J(k, kR) = -dt * dF_xp_dpsi_R / dx;   // contrib. da F_xp

                    // Vicini in z
                    J(k, kD) =  dt * dF_zm_dpsi_D / dz;   // contrib. da F_zm
                    J(k, kU) = -dt * dF_zp_dpsi_U / dz;   // contrib. da F_zp
                }
            }

            // SOLUZIONE del sistema lineare: J * delta_psi = S
            Eigen::VectorXd delta_psi = J.colPivHouseholderQr().solve(S);
            double max_corr = delta_psi.cwiseAbs().maxCoeff();

            psi_new -= delta_psi;

            if (max_corr < tol) break;
        }

        // Update in time: psi^n ← psi^{n+1}
        for (int k = 0; k < Ntot; ++k)
            psi[k] = psi_new(k);

        // Output periodico per animazione
        if (n % 1 == 0) {
            std::ofstream csvout("snapshot_" + std::to_string(n) + ".csv");
            csvout << "x,z,t,psi,theta\n";
            double t = n * dt;
            for (int ix = 0; ix < Nx; ++ix) {
                for (int iz = 0; iz < Nz; ++iz) {
                    int k = idx(ix, iz);
                    double x = ix * dx;
                    double z = iz * dz;
                    csvout << x << "," << z << "," << t << ","
                           << psi[k] << "," << theta(psi[k]) << "\n";
                }
            }
            csvout.close();
        }
    }

    return 0;
}
