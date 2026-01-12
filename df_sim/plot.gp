# plot_eta.gp
# Usage:
#   gnuplot plot_eta.gp
#
# Expected CSV format (with header):
#   x,S,eta
#   ...
# Columns:
#   1 = x, 2 = S(x), 3 = eta(x,t)

set datafile separator ","

set terminal pngcairo size 1200,800 enhanced font "Helvetica,12"
set output "eta_profiles.png"

set title "eta(x,t) with S(x)"
set xlabel "x"
set ylabel "Value"
set grid
set key left top

# If some files are missing, gnuplot will warn but still try to plot others.
# Skip header row with: every ::1

plot \
    "eta_000000.csv" using 1:2 every ::1 with lines lw 2 title "S(x)", \
    "eta_000000.csv" using 1:3 every ::1 with lines lw 2 title "eta (n=0)", \
    "eta_000500.csv" using 1:3 every ::1 with lines lw 2 title "eta (n=500)", \
    "eta_001000.csv" using 1:3 every ::1 with lines lw 2 title "eta (n=1000)", \
    "eta_001500.csv" using 1:3 every ::1 with lines lw 2 title "eta (n=1500)", \
    "eta_002000.csv" using 1:3 every ::1 with lines lw 2 title "eta (n=2000)"

unset output
