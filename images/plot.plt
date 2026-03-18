reset

set key bottom right
set xlabel "array size"
set ylabel "time /ms"

set term svg

set output "graph.svg"
plot "bitonicOrig.txt" w p lt 1 pt 2 t "original", \
     "bitonicMod.txt" w lp lt 2 pt 1 t "modified", \
     "bitonicOrig.txt" w fsteps lt 1 not, \
     "bitonicOrig.txt" w l lt 0 not
     
set output

set logscale x
set logscale y

set output "graph-log.svg"
replot
set output

set xrange[*:1e5]

set output "graph-log-small.svg"
replot
set output

set term x11
replot