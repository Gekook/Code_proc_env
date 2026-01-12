set title 'AMOC Hysteresis: S_tilde vs Forcing F'
set xlabel 'Forcing F (Initial Salinity Difference)'
set ylabel 'S_tilde (= S2 - S1)'
set grid
plot \
  'amoc1var_up.dat' with lines lw 2 lc rgb 'blue' title 'Sweep Up', \
  'amoc1var_down.dat' with lines lw 2 lc rgb 'red' title 'Sweep Down'
pause -1
