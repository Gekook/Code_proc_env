#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

// Parametri modello fisso
const double dt = 1e-5;
const int N = 20000;
const double ro = 10;
const double tau = 10e-3;
const double mus = 1.5;
const double mut = 1.0;

// !!! Cambia solo qui !!!
const double deltaT = 0;     // <--- Prova diversi valori: 0.1, 1, 2, 5, 10, etc
const double Fmin = -0.015;      // <--- Cambia qui il range
const double Fmax = 0.015;       // <--- Cambia qui il range
// !!! Fine delle modifiche consentite !!!

// Funzione modello
double f(double Stilde, double Stildestar, double ro, double tau, double mus, double mut, double deltaT) {
    return (1.0 / tau) * (Stildestar - Stilde) + 2.0 * Stilde * ro * ro * pow((mus * Stilde - mut * deltaT), 2.0);
}

void sweep_smooth(double Fstart, double Fend, double ro, double tau, double mus, double mut, double deltaT, int N_steps, std::ofstream &outfile) {
    double Stilde = 0.0;
    double F = Fstart;
    double Fstep = (Fend - Fstart) / N_steps;
    for (int n = 0; n < N_steps; ++n) {
        double dStdt = f(Stilde, F, ro, tau, mus, mut, deltaT);
        if (std::isnan(dStdt) || std::isinf(dStdt)) break;
        Stilde += dt * dStdt;
        F += Fstep;
        if (std::isnan(Stilde) || std::isinf(Stilde)) break;
        outfile << F << " " << Stilde << "\n";
    }
}

int main() {
    std::ofstream data_up("amoc1var_up.dat");
    sweep_smooth(Fmin, Fmax, ro, tau, mus, mut, deltaT, N, data_up);
    data_up.close();

    std::ofstream data_down("amoc1var_down.dat");
    sweep_smooth(Fmax, Fmin, ro, tau, mus, mut, deltaT, N, data_down);
    data_down.close();

    std::ofstream gp("amoc1var.gnuplot");
    gp << "set title 'AMOC Hysteresis: Stilde vs Forcing F'\n";
    gp << "set xlabel 'Forcing F (Initial Salinity Difference)'\n";
    gp << "set ylabel 'Stilde (= S2 - S1)'\n";
    gp << "set grid\n";
    gp << "plot \\\n";
    gp << "  'amoc1var_up.dat' with lines lw 2 lc rgb 'blue' title 'Sweep Up', \\\n";
    gp << "  'amoc1var_down.dat' with lines lw 2 lc rgb 'red' title 'Sweep Down'\n";
    gp << "pause -1\n";
    gp.close();

    std::cout << "Sto lanciando il plot con gnuplot...\n";
    int status = system("gnuplot amoc1var.gnuplot");
    // Puoi ignorare oppure stampare il codice di stato (0=OK, altro=errore)
    if (status != 0) std::cerr << "Attenzione: gnuplot non è stato eseguito correttamente!\n";

    return 0;
}
