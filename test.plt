set terminal pngcairo dashed enhanced size 640,480
set key outside center top horizontal samplen 3 
set xzeroaxis
set autoscale fix
set ylabel '{/Symbol \341}v{/Symbol \361}' rotate by 360

set output 'fig4.png'
set xlabel 'I_1'
plot 'fig4_bias.dat' using 1:2 title '1^{st}' with lines linewidth 2, 'fig4_bias.dat' using 1:3 title '2^{nd}' with lines linewidth 2
unset output

set output 'fig5a.png'
set xlabel 'I_1'
plot 'fig5a_bias.dat' using 1:2 title '1^{st}' with lines linewidth 2, 'fig5a_bias.dat' using 1:3 title '2^{nd}' with lines linewidth 2
unset output

set output 'fig5b.png'
set logscale x
set format x "10^{%L}"
set xlabel 'D_G'
plot 'fig5b_bias.dat' using 1:2 title '1^{st}' with lines linewidth 2, 'fig5b_bias.dat' using 1:3 title '2^{nd}' with lines linewidth 2
unset output

set output 'fig5c.png'
set xlabel 'D_G'
plot 'fig5c_bias.dat' using 1:2 title '1^{st}' with lines linewidth 2, 'fig5c_bias.dat' using 1:3 title '2^{nd}' with lines linewidth 2
unset output

set output 'fig6a.png'
unset logscale x
set format x "%g"
set xlabel '{/Symbol a}'
plot 'fig6a_bias.dat' using 1:2 title '1^{st}' with lines linewidth 2, 'fig6a_bias.dat' using 1:3 title '2^{nd}' with lines linewidth 2
unset output

set output 'fig6b.png'
set xlabel '{/Symbol a}'
plot 'fig6b_bias.dat' using 1:2 title '1^{st}' with lines linewidth 2, 'fig6b_bias.dat' using 1:3 title '2^{nd}' with lines linewidth 2
unset output

set output 'fig6.8.png'
set xlabel 'I_1'
plot 'fig6.8_bias.dat' using 1:2 title '1^{st}' with lines linewidth 2, 'fig6.8_bias.dat' using 1:3 title '2^{nd}' with lines linewidth 2
unset output

exit gnuplot
