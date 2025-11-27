#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

/**
 * @file amoc.cpp
 * @brief Simple 1D AMOC toy model producing hysteresis data for S_tilde vs F
 *
 * This file integrates a reduced ODE for the state variable S_tilde (S2 - S1)
 * while sweeping the forcing parameter F. It writes upward and downward
 * sweep data files and creates a small gnuplot script to visualize results.
 */

/** @name Fixed model parameters
 *  Constants controlling integration and model coefficients. Avoid changing
 *  these unless you are experimenting deliberately.
 */
/*@{*/

const double ro = 10;          /**< model parameter ro */
const double tau = 1e-3;      /**< timescale tau */
const double mus = 1.5;        /**< parameter mu_s */
const double mut = 1.0;        /**< parameter mu_t */
/*@}*/

/** @name User-configurable parameters
 *  Small set of parameters intended for user experimentation.
 */
/*@{*/


const double dt = 1e-5;
const int    N = 200;
const int    M_steps_time = 10000;
const double deltaT = 3;      
const double S_tilde_star_initial = 0;
const double S_tilde_star_final   = 0.6;
const double tol   = 1e-6;
const double delta = 1e-10;


/** @brief Model right-hand side: computes dS_tilde/dt */
double f(double S_tilde, double S_tilde_star, double ro, double tau, double mus, double mut, double deltaT) {
    return (1.0 / tau) * (S_tilde_star - S_tilde) + 2.0 * S_tilde * ro * ro * pow((mus * S_tilde - mut * deltaT), 2.0);
}

/**
 * @brief Time integrator for S_tilde with fixed forcing F
 *
 * Advances S_tilde starting from S_tilde0 using a forward-Euler scheme for
 * M_steps iterations while keeping forcing F fixed. The function returns the
 * final S_tilde or the last valid value if the computation produces NaN/Inf.
 *
 * @param S_tilde0 Initial S_tilde value
 * @param F Fixed forcing value during integration
 * @param ro Model parameter ro
 * @param tau Timescale parameter
 * @param mus Parameter mu_s
 * @param mut Parameter mu_t
 * @param deltaT Perturbation parameter
 * @param M_steps Number of Euler steps to perform
 * @return Integrated S_tilde after M_steps (or earlier on numerical error)
 */
double integrate_S_tilde(double S_tilde0, double S_tilde_star)
{
    double S_tilde = S_tilde0;
    double S_tilde_old;
    int k = 0;
    double dStdt;

    do {
        S_tilde_old = S_tilde;
        dStdt = f(S_tilde, S_tilde_star, ro, tau, mus, mut, deltaT);
        S_tilde += dt * dStdt;
        ++k;
    } while (k < M_steps_time && std::abs(S_tilde -S_tilde_old) / std::max(std::abs(S_tilde_old), delta) > tol);

    //std::cout<<k<<std::endl;

    return S_tilde;
}

/**
 * @brief Sweep the forcing parameter and record S_tilde for each F
 *
 * For each linearly spaced forcing value between Fstart and Fend this
 * function integrates S_tilde for M_steps_time and writes the pair (F, S_tilde)
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
 * @param outfile Output stream to receive lines "F S_tilde"
 */
void sweep_smooth(double S_tilde_star_initial, double S_tilde_star_final, std::ofstream &outfile) {
    double S_tilde = 0.0; /**< initial S_tilde value */
    double S_tilde_star = S_tilde_star_initial;
    double S_step = (S_tilde_star_final - S_tilde_star_initial) / N;

    for (int n = 0; n < N; ++n) {
        S_tilde = integrate_S_tilde(S_tilde, S_tilde_star);
        outfile << S_tilde_star << " " << S_tilde << "\n";
        S_tilde_star += S_step;
    }
}

int main() {

    std::ofstream data_up("amoc1var_up.dat");
    sweep_smooth(S_tilde_star_initial,S_tilde_star_final, data_up);
    data_up.close();

    std::ofstream data_down("amoc1var_down.dat");
    sweep_smooth(S_tilde_star_final,S_tilde_star_initial,data_down);
    data_down.close();

    std::ofstream gp("amoc1var.gnuplot");
    gp << "set title 'AMOC Hysteresis: S_tilde vs Forcing F'\n";
    gp << "set xlabel 'Forcing F (Initial Salinity Difference)'\n";
    gp << "set ylabel 'S_tilde (= S2 - S1)'\n";
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
