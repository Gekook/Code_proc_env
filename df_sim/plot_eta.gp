set terminal pngcairo size 1200,800 enhanced font 'Helvetica,12'
set output 'eta_profiles.png'
set xlabel 'x'
set ylabel 'Value'
set key left top
set grid
set title 'Hybrid/Implicit scheme: eta(x,t) with S(x)'

plot \
    'eta_000000.csv' using 1:2 with lines lw 2 title 'S(x)', \
    'eta_000000.csv' using 1:3 with lines lw 2 title sprintf('eta (n=%d)',0), \
    'eta_000500.csv' using 1:3 with lines lw 2 title sprintf('eta (n=%d)',500), \
    'eta_001000.csv' using 1:3 with lines lw 2 title sprintf('eta (n=%d)',1000), \
    'eta_001500.csv' using 1:3 with lines lw 2 title sprintf('eta (n=%d)',1500), \
    'eta_002000.csv' using 1:3 with lines lw 2 title sprintf('eta (n=%d)',2000)

# Run with: gnuplot plot_eta.gp
