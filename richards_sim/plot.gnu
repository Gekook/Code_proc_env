### plot_png.gnu
### Ogni file CSV: x,z,t,psi,theta
### Crea un PNG per ogni snapshot con theta e psi affiancati.

reset
set datafile separator ","
set key off
set grid

#-------------------------------------------
# PARAMETRI
#-------------------------------------------
filePattern = "snapshot_%d.csv"    # pattern dei file
tmin = 0                           # primo indice
tmax = 99                           # ultimo indice

#-------------------------------------------
# IMPOSTAZIONI GRAFICHE COMUNI
#-------------------------------------------
set xlabel "x"
set ylabel "z"
set yrange [] reverse              # z verso il basso

set palette rgbformulae 33,13,10
set colorbox

#-------------------------------------------
# LOOP SUGLI SNAPSHOT
#-------------------------------------------
do for [k = tmin:tmax] {

    fname = sprintf(filePattern, k)
    outname = sprintf("snapshot_%d.png", k)

    print sprintf("Creo figura: %s da file %s", outname, fname)

    set terminal pngcairo size 1200,600 enhanced
    set output outname

    set multiplot layout 1,2 title sprintf("Snapshot %d (%s)", k, fname)

        #----------- THETA -----------
        set cblabel "theta"
        set title sprintf("theta(x,z) - snapshot %d", k)
        plot fname using 1:2:5 with image

        #----------- PSI -------------
        set cblabel "psi"
        set title sprintf("psi(x,z) - snapshot %d", k)
        plot fname using 1:2:4 with image

    unset multiplot
    unset output
}

print "Fatto. Sono stati creati i file PNG snapshot_k.png."
