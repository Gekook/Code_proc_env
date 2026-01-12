#include <Eigen/Sparse>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

struct Params {
    int Nx = 401;
    double x0 = -1.0, x1 = 1.0;

    double dt = 1e-4;
    int Nt = 2000;
    int outputEvery = 100;

    double g = 9.81;
    double S = 0.0;
    double theta = 0.35;
    double k0 = 0.01;

    double etaMin = 1e-10;
};

static inline double V_of_eta(double eta, const Params& p) {
    return p.theta * std::max(eta - p.S, p.etaMin);
}

static inline double clamp_eta(double eta, const Params& p) {
    return std::max(eta, p.S + p.etaMin);
}

void writeCSV(const std::string& filename,
              const std::vector<double>& x,
              const std::vector<double>& eta) {
    std::ofstream out(filename);
    out << "x,eta\n";
    for (size_t i = 0; i < x.size(); ++i) out << x[i] << "," << eta[i] << "\n";
}

int main() {
    Params p;

    const int Nx = p.Nx;
    const double dx = (p.x1 - p.x0) / (Nx - 1);

    // Dal modello: dV/deta = theta (costante).
    // alpha_{i+1/2} = (g dt / (theta dx^2)) * (V_{i+1/2}^n * K)
    const double pref = p.g * p.dt / (p.theta * dx * dx);

    std::vector<double> x(Nx), eta(Nx), etaNew(Nx), Vn(Nx), alpha(Nx - 1);

    for (int i = 0; i < Nx; ++i) x[i] = p.x0 + i * dx;

    for (int i = 0; i < Nx; ++i) {
        const double bump = 0.2 * std::exp(-25.0 * x[i] * x[i]);
        eta[i] = clamp_eta(p.S + 1.0 + bump, p);
    }

    writeCSV("eta_000000.csv", x, eta);

    const int N = Nx - 2; // incognite interne: i=1..Nx-2
    Eigen::VectorXd rhs(N), sol(N);

    // Matrice sparsa (tridiagonale) + solver
    Eigen::SparseMatrix<double> A(N, N);
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

    for (int n = 1; n <= p.Nt; ++n) {

        for (int i = 0; i < Nx; ++i) {
            eta[i] = clamp_eta(eta[i], p);
            Vn[i] = V_of_eta(eta[i], p);
        }

        for (int i = 0; i < Nx - 1; ++i) {
            const double Vhalf = 0.5 * (Vn[i] + Vn[i + 1]);
            alpha[i] = pref * (Vhalf * p.k0);
        }

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(3 * N);

        for (int j = 0; j < N; ++j) {
            const int i = j + 1;              // nodo fisico i
            const double a_i = alpha[i - 1];  // alpha_{i-1/2}
            const double c_i = alpha[i];      // alpha_{i+1/2}

            double diag = 1.0 + a_i + c_i;
            double sub  = -a_i;
            double sup  = -c_i;

            // NO-FLUX: eta_0 = eta_1 e eta_{Nx-1} = eta_{Nx-2}
            // Prima equazione interna (i=1): il termine sub*eta_0 diventa sub*eta_1 => aggiunge a diagonale.
            if (j == 0) {
                diag += sub; // sub è negativo: diag += (-a_i) equivale a b[0] -= a[0] in Thomas
                sub = 0.0;
            }
            // Ultima equazione interna (i=Nx-2): il termine sup*eta_{Nx-1} diventa sup*eta_{Nx-2} => aggiunge a diagonale.
            if (j == N - 1) {
                diag += sup;
                sup = 0.0;
            }

            trips.emplace_back(j, j, diag);
            if (j > 0)     trips.emplace_back(j, j - 1, sub);
            if (j < N - 1) trips.emplace_back(j, j + 1, sup);

            rhs[j] = eta[i]; // RHS = eta_i^n
        }

        A.setZero();
        A.setFromTriplets(trips.begin(), trips.end());
        A.makeCompressed();

        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Eigen::SparseLU factorization failed at step " << n << "\n";
            return 1;
        }

        sol = solver.solve(rhs);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Eigen::SparseLU solve failed at step " << n << "\n";
            return 1;
        }

        for (int j = 0; j < N; ++j) etaNew[j + 1] = clamp_eta(sol[j], p);

        // NO-FLUX (gradiente nullo): eta_0 = eta_1, eta_{Nx-1} = eta_{Nx-2}
        etaNew[0] = etaNew[1];
        etaNew[Nx - 1] = etaNew[Nx - 2];

        eta.swap(etaNew);

        if (n % p.outputEvery == 0) {
            char name[64];
            std::snprintf(name, sizeof(name), "eta_%06d.csv", n);
            writeCSV(name, x, eta);
            std::cout << "Saved " << name << "\n";
        }
    }

    return 0;
}
