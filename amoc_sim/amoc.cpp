#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

/**
 * @file amoc.cpp
 * @brief Simple 1D AMOC toy model producing hysteresis data for Stilde vs F
 *
 * This file integrates a reduced ODE for the state variable Stilde (S2 - S1)
 * while sweeping the forcing parameter F. It writes upward and downward
 * sweep data files and creates a small gnuplot script to visualize results.
 */

/** @name Fixed model parameters
 *  Constants controlling integration and model coefficients. Avoid changing
 *  these unless you are experimenting deliberately.
 */
/*@{*/
const double dt = 1e-5;        /**< time step for Euler integration */
const int N = 40;            /**< number of sweep steps */
const int M_steps_time = 100;  /**< temporal integration steps per F value */
const double ro = 10;          /**< model parameter ro */
const double tau = 10e-3;      /**< timescale tau */
const double mus = 1.5;        /**< parameter mu_s */
const double mut = 1.0;        /**< parameter mu_t */
/*@}*/

/** @name User-configurable parameters
 *  Small set of parameters intended for user experimentation.
 */
/*@{*/
const double deltaT = 0;       /**< perturbation parameter (try non-zero values) */
const double Fmin = -0.0015;    /**< minimum forcing for sweep */
const double Fmax = 0.0015;     /**< maximum forcing for sweep */
/*@}*/

/** @brief Model right-hand side: computes dStilde/dt */
double f(double Stilde, double Stildestar, double ro, double tau, double mus, double mut, double deltaT) {
    return (1.0 / tau) * (Stildestar - Stilde) + 2.0 * Stilde * ro * ro * pow((mus * Stilde - mut * deltaT), 2.0);
}

/**
 * @brief Time integrator for Stilde with fixed forcing F
 *
 * Advances Stilde starting from Stilde0 using a forward-Euler scheme for
 * M_steps iterations while keeping forcing F fixed. The function returns the
 * final Stilde or the last valid value if the computation produces NaN/Inf.
 *
 * @param Stilde0 Initial Stilde value
 * @param F Fixed forcing value during integration
 * @param ro Model parameter ro
 * @param tau Timescale parameter
 * @param mus Parameter mu_s
 * @param mut Parameter mu_t
 * @param deltaT Perturbation parameter
 * @param M_steps Number of Euler steps to perform
 * @return Integrated Stilde after M_steps (or earlier on numerical error)
 */
double integrate_Stilde(double Stilde0, double F, double ro, double tau, double mus, double mut, double deltaT, int M_steps) {
    double Stilde = Stilde0;
    for (int k = 0; k < M_steps; ++k) {
        double dStdt = f(Stilde, F, ro, tau, mus, mut, deltaT);
        if (std::isnan(dStdt) || std::isinf(dStdt)) break;
        Stilde += dt * dStdt;
        if (std::isnan(Stilde) || std::isinf(Stilde)) break;
    }
    return Stilde;
}

/**
 * @brief Sweep the forcing parameter and record Stilde for each F
 *
 * For each linearly spaced forcing value between Fstart and Fend this
 * function integrates Stilde for M_steps_time and writes the pair (F, Stilde)
 * to the provided output stream.
 *
 * @param Fstart Starting forcing value
 * @param Fend Ending forcing value
 * @param ro Model parameter ro (passed through)
 * @param tau Timescale parameter (passed through)
 * @param mus Parameter mu_s (passed through)
 * @param mut Parameter mu_t (passed through)
 * @param deltaT Perturbation parameter (passed through)
 * @param N_steps_sweep Number of forcing steps in the sweep
 * @param M_steps_time Number of temporal integration steps per forcing
 * @param outfile Output stream to receive lines "F Stilde"
 */
void sweep_smooth(double Fstart, double Fend, double ro, double tau, double mus, double mut, double deltaT,
                  int N_steps_sweep, int M_steps_time, std::ofstream &outfile) {
    double Stilde = 0.0; /**< initial Stilde value */
    double F = Fstart;
    double Fstep = (Fend - Fstart) / N_steps_sweep;

    for (int n = 0; n < N_steps_sweep; ++n) {
        Stilde = integrate_Stilde(Stilde, F, ro, tau, mus, mut, deltaT, M_steps_time);
        outfile << F << " " << Stilde << "\n";
        F += Fstep;
    }
}

int main() {
    std::ofstream data_up("amoc1var_up.dat");
    sweep_smooth(Fmin, Fmax, ro, tau, mus, mut, deltaT, N, M_steps_time, data_up);
    data_up.close();

    std::ofstream data_down("amoc1var_down.dat");
    sweep_smooth(Fmax, Fmin, ro, tau, mus, mut, deltaT, N, M_steps_time, data_down);
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

    std::cout << "Execution of gnuplot...\n";
    int status = system("gnuplot amoc1var.gnuplot");
    if (status != 0) std::cerr << "Attention: gnuplot was not executed correctly!\n";

    return 0;
}
