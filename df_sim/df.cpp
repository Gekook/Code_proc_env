#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

struct Params {
    // 1D spatial domain
    int Nx = 401;
    double x0 = -1.0, x1 = 1.0;

    // Time discretization
    double dt = 1e-4;
    int Nt = 2000;
    int outputEvery = 100;

    // Physical parameters
    double g = 9.81;
    double S = 0.0;          // Bedrock elevation
    double etaMin = 1e-10;   // Minimum water height above S

    // Vertical quadrature for integrals
    int NzInt = 80;

    // Boundary conditions
    bool noFluxBC = true;
};

/*------------------------------------------------------------
  Vertical profiles
  Modify these according to the physical model
------------------------------------------------------------*/

// Big Theta(z)
static inline double Theta(double z) {
    (void)z;
    return 0.35;
}

// k tilde(z)
static inline double kTilde(double z) {
    (void)z;
    return 1e-2;
}

/*------------------------------------------------------------
  V(eta) = ∫_S^eta Theta(z) dz
  dV/deta = Theta(eta)
------------------------------------------------------------*/

double V(double eta, const Params& p) {
    if (eta <= p.S) return p.etaMin;

    const int n = std::max(1, p.NzInt);
    const double dz = (eta - p.S) / n;

    double integral = 0.0;
    for (int k = 0; k <= n; ++k) {
        const double z = p.S + k * dz;
        const double w = (k == 0 || k == n) ? 0.5 : 1.0;
        integral += w * Theta(z);
    }
    integral *= dz;

    return std::max(integral, p.etaMin);
}

double dVdEta(double eta) {
    return Theta(eta);
}

/*------------------------------------------------------------
  Hydraulic conductivity
  K(eta) = (1 / V(eta)) ∫_S^eta Theta(z) k~(z) dz
------------------------------------------------------------*/

double K_of_eta(double eta, const Params& p) {
    if (eta <= p.S) return 0.0;

    const int n = std::max(1, p.NzInt);
    const double dz = (eta - p.S) / n;

    double integral = 0.0;
    for (int k = 0; k <= n; ++k) {
        const double z = p.S + k * dz;
        const double w = (k == 0 || k == n) ? 0.5 : 1.0;
        integral += w * Theta(z) * kTilde(z);
    }
    integral *= dz;

    return integral / V(eta, p);
}

/*------------------------------------------------------------
  Output utility
------------------------------------------------------------*/

void writeCSV(const std::string& filename,
              const std::vector<double>& x,
              const std::vector<double>& eta) {
    std::ofstream out(filename);
    out << "x,eta\n";
    for (size_t i = 0; i < x.size(); ++i)
        out << x[i] << "," << eta[i] << "\n";
}

/*------------------------------------------------------------
  Main solver
------------------------------------------------------------*/

int main() {
    Params p;

    const int Nx = p.Nx;
    const double dx = (p.x1 - p.x0) / (Nx - 1);

    std::vector<double> x(Nx), eta(Nx), etaNew(Nx);
    std::vector<double> flux(Nx - 1);

    // Spatial grid
    for (int i = 0; i < Nx; ++i)
        x[i] = p.x0 + i * dx;

    // Initial condition
    for (int i = 0; i < Nx; ++i) {
        double bump = 0.2 * std::exp(-25.0 * x[i] * x[i]);
        eta[i] = p.S + 1.0 + bump;
    }

    writeCSV("eta_000000.csv", x, eta);

    // Time loop
    for (int n = 1; n <= p.Nt; ++n) {

        // Flux at cell interfaces (i+1/2)
        for (int i = 0; i < Nx - 1; ++i) {
            double Vi  = V(eta[i], p);
            double Vip = V(eta[i + 1], p);

            double Ki  = K_of_eta(eta[i], p);
            double Kip = K_of_eta(eta[i + 1], p);

            double VK_half = 0.5 * (Vi * Ki + Vip * Kip);
            double gradEta = (eta[i + 1] - eta[i]) / dx;

            flux[i] = p.g * VK_half * gradEta;
        }

        // Explicit Euler update
        for (int i = 1; i < Nx - 1; ++i) {
            double RHS = (flux[i] - flux[i - 1]) / dx;
            etaNew[i] = eta[i] + p.dt * RHS / dVdEta(eta[i]);
            etaNew[i] = std::max(etaNew[i], p.S + p.etaMin);
        }

        // Boundary conditions
        if (p.noFluxBC) {
            etaNew[0]    = etaNew[1];
            etaNew[Nx-1] = etaNew[Nx-2];
        }

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
