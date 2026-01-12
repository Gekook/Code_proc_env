#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Params {
    int Nx = 401;
    double x0 = -1.0, x1 = 1.0;

    double dt = 1e-4;
    int Nt = 2000;
    int outputEvery = 100;

    double g = 9.81;
    double theta = 0.35;
    double k0 = 0.01;

    // eta >= S(x) + etaMin
    double etaMin = 1e-10;

    // Optional: load S(x) from CSV (x,S). If empty -> use analytic S(x).
    std::string S_profile_csv = ""; // e.g. "S_profile.csv"

    // Steps to include in the gnuplot script (must match output files).
    std::vector<int> plotSteps = {0, 500, 1000, 1500, 2000};

    // Output gnuplot script name
    std::string gnuplotScript = "plot_eta.gp";
    std::string gnuplotFigure = "eta_profiles.png";
};

// -------------------- S(x) handling --------------------

struct Tabulated1D {
    std::vector<double> xs, ys;

    bool loadCSV(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) return false;

        xs.clear(); ys.clear();
        std::string line;
        // allow optional header
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            if (line.find_first_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") != std::string::npos) {
                // skip header-like lines
                continue;
            }
            std::stringstream ss(line);
            std::string a, b;
            if (!std::getline(ss, a, ',')) continue;
            if (!std::getline(ss, b, ',')) continue;
            xs.push_back(std::stod(a));
            ys.push_back(std::stod(b));
        }
        // ensure sorted
        std::vector<size_t> idx(xs.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j){ return xs[i] < xs[j]; });

        std::vector<double> xs2, ys2;
        xs2.reserve(xs.size()); ys2.reserve(ys.size());
        for (auto i : idx) { xs2.push_back(xs[i]); ys2.push_back(ys[i]); }
        xs.swap(xs2); ys.swap(ys2);

        return !xs.empty();
    }

    double eval(double x) const {
        if (xs.empty()) return 0.0;
        if (x <= xs.front()) return ys.front();
        if (x >= xs.back())  return ys.back();

        auto it = std::upper_bound(xs.begin(), xs.end(), x);
        size_t j = size_t(it - xs.begin());
        size_t i = j - 1;
        double x0 = xs[i], x1 = xs[j];
        double y0 = ys[i], y1 = ys[j];
        double t = (x - x0) / (x1 - x0);
        return (1.0 - t) * y0 + t * y1;
    }
};

static inline double S_analytic(double x) {
    const double S0 = 0.15;
    const double H1 = 0.25;
    const double H2 = 0.20;
    const double w  = 0.05;

    // SPECULARE: usa -x
    double step1 = 0.5 * (1.0 + std::tanh((-x + 0.4) / w));
    double step2 = 0.5 * (1.0 + std::tanh((-x - 0.1) / w));

    return S0 + H1 * step1 + H2 * step2;
}



static inline double clamp_eta(double eta, double Sx, const Params& p) {
    return std::max(eta, Sx + p.etaMin);
}

static inline double V_of_eta(double eta, double Sx, const Params& p) {
    // V(eta) = theta * max(eta - S(x), etaMin)
    return p.theta * std::max(eta - Sx, p.etaMin);
}

// -------------------- I/O --------------------

void writeCSV(const std::string& filename,
              const std::vector<double>& x,
              const std::vector<double>& Sx,
              const std::vector<double>& eta) {
    std::ofstream out(filename);
    out << "x,S,eta\n";
    for (size_t i = 0; i < x.size(); ++i) {
        out << x[i] << "," << Sx[i] << "," << eta[i] << "\n";
    }
}

static std::string stepFilename(int n) {
    char name[64];
    std::snprintf(name, sizeof(name), "eta_%06d.csv", n);
    return std::string(name);
}

void writeGnuplotScript(const Params& p) {
    std::ofstream gp(p.gnuplotScript);
    gp << "set terminal pngcairo size 1200,800 enhanced font 'Helvetica,12'\n";
    gp << "set output '" << p.gnuplotFigure << "'\n";
    gp << "set xlabel 'x'\n";
    gp << "set ylabel 'Value'\n";
    gp << "set key left top\n";
    gp << "set grid\n";
    gp << "set title 'Hybrid/Implicit scheme: eta(x,t) with S(x)'\n\n";

    // Plot eta at selected steps + S(x) from the first file (column 2)
    // Columns: 1=x, 2=S, 3=eta
    gp << "plot \\\n";
    gp << "    '" << stepFilename(0) << "' using 1:2 with lines lw 2 title 'S(x)', \\\n";

    for (size_t k = 0; k < p.plotSteps.size(); ++k) {
        int n = p.plotSteps[k];
        gp << "    '" << stepFilename(n) << "' using 1:3 with lines lw 2 title sprintf('eta (n=%d)'," << n << ")";
        gp << (k + 1 < p.plotSteps.size() ? ", \\\n" : "\n");
    }

    gp << "\n# Run with: gnuplot " << p.gnuplotScript << "\n";
}

// -------------------- Main --------------------

int main() {
    Params p;

    const int Nx = p.Nx;
    const double dx = (p.x1 - p.x0) / (Nx - 1);

    // pref = g dt / (theta dx^2)
    const double pref = p.g * p.dt / (p.theta * dx * dx);

    std::vector<double> x(Nx), Sx(Nx), eta(Nx), etaNew(Nx), Vn(Nx), alpha(Nx - 1);

    for (int i = 0; i < Nx; ++i) x[i] = p.x0 + i * dx;

    // Build S(x): from CSV if provided, else analytic.
    Tabulated1D S_tab;
    bool useTab = false;
    if (!p.S_profile_csv.empty()) {
        useTab = S_tab.loadCSV(p.S_profile_csv);
        if (!useTab) {
            std::cerr << "Warning: could not read '" << p.S_profile_csv
                      << "'. Falling back to analytic S(x).\n";
        }
    }
    for (int i = 0; i < Nx; ++i) {
        Sx[i] = useTab ? S_tab.eval(x[i]) : S_analytic(x[i]);
    }

    const double xf = -0.6;   // fronte iniziale
    const double H  = 0.35;   // livello sopra S
    const double w  = 0.04;   // smoothing

    for (int i = 0; i < Nx; ++i) {
        double step = 0.5 * (1.0 - std::tanh((x[i] - xf) / w));
        eta[i] = clamp_eta(Sx[i] + H * step, Sx[i], p);
    }


    // Output initial
    writeCSV(stepFilename(0), x, Sx, eta);

    const int N = Nx - 2; // internal unknowns i=1..Nx-2
    Eigen::VectorXd rhs(N), sol(N);

    Eigen::SparseMatrix<double> A(N, N);
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

    for (int n = 1; n <= p.Nt; ++n) {
        // Update V(eta) using local S(x)
        for (int i = 0; i < Nx; ++i) {
            eta[i] = clamp_eta(eta[i], Sx[i], p);
            Vn[i]  = V_of_eta(eta[i], Sx[i], p);
        }

        // alpha_{i+1/2} uses V at half-step (average)
        for (int i = 0; i < Nx - 1; ++i) {
            const double Vhalf = 0.5 * (Vn[i] + Vn[i + 1]);
            alpha[i] = pref * (Vhalf * p.k0);
        }

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(3 * N);

        for (int j = 0; j < N; ++j) {
            const int i = j + 1;              // physical node i
            const double a_i = alpha[i - 1];  // alpha_{i-1/2}
            const double c_i = alpha[i];      // alpha_{i+1/2}

            double diag = 1.0 + a_i + c_i;
            double sub  = -a_i;
            double sup  = -c_i;

            // NO-FLUX: eta_0 = eta_1 and eta_{Nx-1} = eta_{Nx-2}
            if (j == 0) {
                diag += sub;
                sub = 0.0;
            }
            if (j == N - 1) {
                diag += sup;
                sup = 0.0;
            }

            trips.emplace_back(j, j, diag);
            if (j > 0)     trips.emplace_back(j, j - 1, sub);
            if (j < N - 1) trips.emplace_back(j, j + 1, sup);

            rhs[j] = eta[i];
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

        for (int j = 0; j < N; ++j) etaNew[j + 1] = clamp_eta(sol[j], Sx[j + 1], p);

        // NO-FLUX boundary
        etaNew[0]      = etaNew[1];
        etaNew[Nx - 1] = etaNew[Nx - 2];

        eta.swap(etaNew);

        if (n % p.outputEvery == 0) {
            writeCSV(stepFilename(n), x, Sx, eta);
            std::cout << "Saved " << stepFilename(n) << "\n";
        }
    }

    // Write a gnuplot script to plot selected snapshots (and S(x))
    writeGnuplotScript(p);
    std::cout << "Saved gnuplot script: " << p.gnuplotScript << "\n";
    std::cout << "Run: gnuplot " << p.gnuplotScript << "\n";

    return 0;
}
