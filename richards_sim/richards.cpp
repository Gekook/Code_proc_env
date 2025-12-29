// richards_clone_python_with_csv.cpp
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    // ----------------------------
    // Paramètres physiques / numériques (IDENTIQUES au Python)
    // ----------------------------
    const double C = 1.0;
    const double m = 2.0;
    const double K0 = 3e-3;
    const double g = -10.0;

    const double L = 0.1;
    const int Nz = 301;
    const double dz = L / (Nz - 1);

    const double dt = 5e-3;
    const double t_final = 2.8;
    const int nt = (int)std::ceil(t_final / dt);

    const double q_in = 1e-3; // not used because TOP flux branch is commented in Python

    const double tol_newton = 1e-4;
    const int max_newton = 20;

    const double Theta_max = 0.35;

    // ----------------------------
    // OUTPUT SETTINGS (NEW)
    // ----------------------------
    const int output_every = 10;                 // dump every N steps (like you want)
    const std::string out_prefix = "richards_";  // produces richards_00010.csv, ...

    auto write_csv_snapshot = [&](int step, double t, const Eigen::VectorXd& psi,
                                  const Eigen::VectorXd& theta) {
        std::ostringstream name;
        name << out_prefix << std::setw(5) << std::setfill('0') << step << ".csv";
        std::ofstream f(name.str());
        if (!f) {
            std::cerr << "Cannot open output file: " << name.str() << "\n";
            return;
        }

        // ParaView-friendly: header + columns
        // You can use "Table To Points": X=z, Y=0, Z=0, and scalar=psi/theta
        f << "t,z,psi,theta\n";
        f << std::setprecision(17);

        for (int i = 0; i < Nz; ++i) {
            const double z = i * dz;
            f << t << "," << z << "," << psi[i] << "," << theta[i] << "\n";
        }
    };

    // ----------------------------
    // Lois constitutives (IDENTIQUES)
    // ----------------------------
    auto theta_of_psi = [&](const Eigen::VectorXd& psi) {
        Eigen::VectorXd out(Nz);
        for (int i = 0; i < Nz; ++i) {
            const double psi_pos = std::max(psi[i], 0.0);
            const double theta_unsat = std::pow(psi_pos / C, 1.0 / m);
            out[i] = theta_unsat; // EXACT like Python (no min with Theta_max here)
        }
        return out;
    };

    auto dtheta_dpsi = [&](const Eigen::VectorXd& psi) {
        Eigen::VectorXd deriv(Nz);
        for (int i = 0; i < Nz; ++i) {
            const double psi_pos = std::max(psi[i], 1e-16);
            const double theta_unsat = std::pow(psi_pos / C, 1.0 / m);

            double d = (1.0 / m) * std::pow(1.0 / C, 1.0 / m) * std::pow(psi_pos, 1.0 / m - 1.0);

            if (theta_unsat >= Theta_max) d = 0.0;
            if (psi[i] < 0.0) d = 0.0;

            deriv[i] = d;
        }
        return deriv;
    };

    // ----------------------------
    // Condition initiale (IDENTIQUE)
    // ----------------------------
    const double theta_init_top = 0.2;
    const double theta_init_bottom = 0.0;

    Eigen::VectorXd theta0 = Eigen::VectorXd::Zero(Nz);
    theta0[0] = theta_init_top;
    for (int k = 1; k < Nz; ++k) {
        double zfrac = (double)k / (Nz - 1);
        (void)zfrac;
        theta0[k] = 0.0; // EXACT Python line: theta0[k] = 0 # ...
    }

    Eigen::VectorXd psi0(Nz);
    for (int i = 0; i < Nz; ++i) psi0[i] = C * std::pow(theta0[i], m);
    for (int i = 0; i < Nz; ++i) {
        if (theta0[i] >= Theta_max) psi0[i] = C * std::pow(Theta_max, m);
    }

    // ----------------------------
    // Conditions aux limites (IDENTIQUES)
    // ----------------------------
    const double theta_top_imposed = 0.2;
    const double psi_top = C * std::pow(theta_top_imposed, m);

    // ----------------------------
    // Conductivité (IDENTIQUE)
    // ----------------------------
    Eigen::VectorXd K_nodes = Eigen::VectorXd::Constant(Nz, K0);

    auto K_face = [&](const Eigen::VectorXd& K_nodes_in) {
        Eigen::VectorXd Kf(Nz - 1);
        for (int k = 0; k < Nz - 1; ++k) {
            Kf[k] = std::sqrt(K_nodes_in[k] * K_nodes_in[k + 1]);
        }
        return Kf;
    };

    // ----------------------------
    // build_S_and_J (IDENTIQUE AU PYTHON)
    // ----------------------------
    auto build_S_and_J = [&](const Eigen::VectorXd& psi_nplus1,
                            const Eigen::VectorXd& psi_n,
                            Eigen::VectorXd& S_out,
                            Eigen::SparseMatrix<double>& J_out) {
        Eigen::VectorXd Kf = K_face(K_nodes);

        Eigen::VectorXd theta_n = theta_of_psi(psi_n);

        Eigen::VectorXd theta_half = Eigen::VectorXd::Zero(Nz - 1);
        for (int k = 0; k < Nz - 1; ++k) {
            theta_half[k] = 0.5 * (theta_n[k + 1] + theta_n[k]);
        }

        Eigen::VectorXd diag = Eigen::VectorXd::Zero(Nz);
        Eigen::VectorXd off_lo = Eigen::VectorXd::Zero(Nz - 1);
        Eigen::VectorXd off_hi = Eigen::VectorXd::Zero(Nz - 1);

        Eigen::VectorXd dtheta_nplus1 = dtheta_dpsi(psi_nplus1);
        Eigen::VectorXd theta_nplus1 = theta_of_psi(psi_nplus1);

        Eigen::VectorXd S = Eigen::VectorXd::Zero(Nz);

        // Intérieur: k=1..Nz-2
        for (int k = 1; k < Nz - 1; ++k) {
            const double Kkp = Kf[k];
            const double Kkm = Kf[k - 1];
            const double thp = theta_half[k];
            const double thm = theta_half[k - 1];

            const double F =
                (Kkp * thp * (psi_nplus1[k + 1] - psi_nplus1[k]) / dz
               - Kkm * thm * (psi_nplus1[k]     - psi_nplus1[k - 1]) / dz);

            const double grav = g * Kkp * (theta_n[k] - theta_n[k - 1]);

            S[k] = theta_nplus1[k] - theta_n[k] - (dt / dz) * (F + grav);

            diag[k] = dtheta_nplus1[k] + (dt / dz) * (Kkp * thp + Kkm * thm) / dz;
            off_hi[k] = -(dt / dz) * Kkp * thp / dz;
            off_lo[k - 1] = -(dt / dz) * Kkm * thm / dz;
        }

        // Top: Dirichlet (active branch)
        S[0] = psi_nplus1[0] - psi_top;
        diag[0] = 1.0;
        off_hi[0] = 0.0;

        // Bottom: "Neumann flux nul" implemented like Python residual
        {
            int k = Nz - 1;
            const double Kkm = Kf[Nz - 2];
            const double thm = theta_half[Nz - 2];

            const double F = -Kkm * thm * (psi_nplus1[k] - psi_nplus1[k - 1]) / dz;

            S[k] = theta_nplus1[k] - theta_n[k] - (dt / dz) * F;

            diag[k] = dtheta_nplus1[k] + (dt / dz) * Kkm * thm / dz;
            off_lo[k - 1] = -(dt / dz) * Kkm * thm / dz;
        }

        // Build sparse tri-diagonal J
        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve(3 * Nz);

        for (int i = 0; i < Nz; ++i) {
            trip.emplace_back(i, i, diag[i]);
            if (i < Nz - 1) trip.emplace_back(i, i + 1, off_hi[i]);
            if (i > 0)      trip.emplace_back(i, i - 1, off_lo[i - 1]);
        }

        J_out.resize(Nz, Nz);
        J_out.setFromTriplets(trip.begin(), trip.end());

        S_out = S;
    };

    // ----------------------------
    // Boucle en temps (IDENTIQUE)
    // ----------------------------
    Eigen::VectorXd psi_n = psi0;

    // Dump initial snapshot
    {
        Eigen::VectorXd theta_now = theta_of_psi(psi_n);
        write_csv_snapshot(0, 0.0, psi_n, theta_now);
    }

    std::cout << std::setprecision(10);
    for (int n = 0; n < nt; ++n) {
        const double t = (n + 1) * dt;
        Eigen::VectorXd psi_np1 = psi_n;

        for (int it = 0; it < max_newton; ++it) {
            Eigen::VectorXd S(Nz);
            Eigen::SparseMatrix<double> J;

            build_S_and_J(psi_np1, psi_n, S, J);

            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.analyzePattern(J);
            solver.factorize(J);

            if (solver.info() != Eigen::Success) {
                std::cerr << "SparseLU factorization failed at t=" << t << "\n";
                break;
            }

            Eigen::VectorXd dpsi = solver.solve(S);
            if (solver.info() != Eigen::Success) {
                std::cerr << "SparseLU solve failed at t=" << t << "\n";
                break;
            }

            psi_np1 -= dpsi;

            const double inf_norm = dpsi.cwiseAbs().maxCoeff();
            if (inf_norm < tol_newton) break;
        }

        psi_n = psi_np1;

        Eigen::VectorXd theta_now = theta_of_psi(psi_n);
        const double theta_min = theta_now.minCoeff();
        const double theta_max = theta_now.maxCoeff();

        std::cout << "t=" << std::fixed << std::setprecision(3) << t
                  << " s: min theta=" << std::setprecision(4) << theta_min
                  << ", max theta=" << theta_max << "\n";

        // CSV snapshot every output_every
        if ((n + 1) % output_every == 0 || (n == nt - 1)) {
            write_csv_snapshot(n + 1, t, psi_n, theta_now);
        }
    }

    return 0;
}
