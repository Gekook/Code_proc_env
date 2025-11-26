/**
 * @file amoc.cpp
 * @brief Simple 1D AMOC (Atlantic Meridional Overturning Circulation) toy model
 *
 * This program integrates a simple ordinary differential equation for the
 * variable Stilde (salinity difference S2 - S1) while sweeping an external
 * forcing parameter F. It writes two data files for upward and downward
 * sweeps and generates a small gnuplot script to visualize hysteresis.
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

/**
 * @brief Fixed model parameters
 *
 * These constants control time-stepping and the model coefficients. They are
 * declared as constants and should not be modified by the rest of the code.
 */
const double dt = 1e-5;    /**< time step for Euler integration */
const int N = 20000;       /**< number of sweep steps */
const double ro = 10;      /**< physical parameter ro (model constant) */
const double tau = 10e-3;  /**< timescale tau */
const double mus = 1.5;    /**< parameter mu_s */
const double mut = 1.0;    /**< parameter mu_t */

/**
 * @brief User-configurable region
 *
 * These variables are intended to be changed for experimentation. Do not alter
 * code logic — only modify the numeric values if you want to explore different
 * behaviors.
 */
const double deltaT = 0;     /**< perturbation parameter (try values like 0.1, 1, 2) */
const double Fmin = -0.015;  /**< lower bound of forcing sweep */
const double Fmax = 0.015;   /**< upper bound of forcing sweep */

/**
 * @brief Right-hand side of the ODE for Stilde
 *
 * Computes dStilde/dt according to a reduced AMOC toy model.
 *
 * @param Stilde Current value of the state variable Stilde (S2 - S1)
 * @param Stildestar Forcing parameter that acts like a target Stilde (F)
 * @param ro Model parameter ro
 * @param tau Timescale parameter tau
 * @param mus Parameter mu_s
 * @param mut Parameter mu_t
 * @param deltaT Perturbation parameter
 * @return The time derivative dStilde/dt
 */
double f(double Stilde, double Stildestar, double ro, double tau, double mus, double mut, double deltaT) {
    return (1.0 / tau) * (Stildestar - Stilde) + 2.0 * Stilde * ro * ro * pow((mus * Stilde - mut * deltaT), 2.0);
}

/**
 * @brief Perform a smooth sweep of the forcing parameter and integrate the ODE
 *
 * This function integrates the ODE using a simple forward Euler step while
 * linearly changing the forcing from Fstart to Fend in N_steps. Results are
 * written to the provided output stream as pairs of (F, Stilde) per line.
 *
 * @param Fstart Starting forcing value for the sweep
 * @param Fend Ending forcing value for the sweep
 * @param ro Model parameter ro (passed through)
 * @param tau Timescale parameter tau (passed through)
 * @param mus Parameter mu_s (passed through)
 * @param mut Parameter mu_t (passed through)
 * @param deltaT Perturbation parameter (passed through)
 * @param N_steps Number of integration/sweep steps
 * @param outfile Output stream where results are appended (format: F Stilde) 
 */
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

/**
 * @brief Program entry point
 *
 * Generates two data files for upward and downward sweeps and creates a small
 * gnuplot script to visualize Stilde vs. forcing F. The program tries to call
 * gnuplot to display the result; a non-zero return code is reported on stderr.
 *
 * @return 0 on success
 */
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
    gp << "plot \\\n+";
    gp << "  'amoc1var_up.dat' with lines lw 2 lc rgb 'blue' title 'Sweep Up', \\\n+";
    gp << "  'amoc1var_down.dat' with lines lw 2 lc rgb 'red' title 'Sweep Down'\n";
    gp << "pause -1\n";
    gp.close();

    std::cout << "Execution of gnuplot...\n";
    int status = system("gnuplot amoc1var.gnuplot");
    // You may ignore or print the exit status (0=OK, non-zero=error)
    if (status != 0) std::cerr << "Attention: gnuplot was not executed correctly!\n";

    return 0;
}
