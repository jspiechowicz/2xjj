#!/usr/bin/python
import commands, os
import numpy

#Model
amp = 1.8912 
omega = 0.2708
force = 0
alpha = 0.77
Dg = 5.0e-05
Dp = 0
lmd = 0
comp = 0

#Simulation
dev = 0
block = 64
paths = 1024
periods = 1000
spp = 200
algorithm = 'predcorr'
trans = 0.1

#Output
mode = 'moments'
points = 100
beginx = 0
endx = 0.04
domain = '1d'
domainx = 'f'
logx = 0
DIRNAME='./tests/2xjj/'
os.system('mkdir -p %s' % DIRNAME)
os.system('rm -v %s*.dat %s*.png' % (DIRNAME, DIRNAME))

#fig 4

outb = 'fig4_bias'
_cmd = './prog --dev=%d --amp=%s --omega=%s --force=%s --alpha=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --block=%d --paths=%d --periods=%s --spp=%d --algorithm=%s --trans=%s --mode=%s --points=%d --beginx=%s --endx=%s --domain=%s --domainx=%s --logx=%d >> %s.dat' % (dev, amp, omega, force, alpha, Dg, Dp, lmd, comp, block, paths, periods, spp, algorithm, trans, mode, points, beginx, endx, domain, domainx, logx, outb)
output = open('%s.dat' % outb, 'w')
print >>output, '#%s' % _cmd
output.close()
print _cmd
cmd = commands.getoutput(_cmd)

#fig 5a

amp = 1.7754
omega = 0.1876
force = 0
alpha = 0.77
Dg = 2.0e-05
Dp = 0
lmd = 0
domainx = 'f'
endx = 0.06

outb = 'fig5a_bias'
_cmd = './prog --dev=%d --amp=%s --omega=%s --force=%s --alpha=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --block=%d --paths=%d --periods=%s --spp=%d --algorithm=%s --trans=%s --mode=%s --points=%d --beginx=%s --endx=%s --domain=%s --domainx=%s --logx=%d >> %s.dat' % (dev, amp, omega, force, alpha, Dg, Dp, lmd, comp, block, paths, periods, spp, algorithm, trans, mode, points, beginx, endx, domain, domainx, logx, outb)
output = open('%s.dat' % outb, 'w')
print >>output, '#%s' % _cmd
output.close()
print _cmd
cmd = commands.getoutput(_cmd)

#fig 5b, 5c

amp = 1.7754
omega = 0.1876
force = 0.025
alpha = 0.77
Dg = 0
domainx = 'D'
logx = 1
beginx = -6
endx = -2

for force in [0.025, 0.046]:
    if force == 0.025:
        outb = 'fig5b_bias'
    elif force == 0.046:
        outb = 'fig5c_bias'
    _cmd = './prog --dev=%d --amp=%s --omega=%s --force=%s --alpha=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --block=%d --paths=%d --periods=%s --spp=%d --algorithm=%s --trans=%s --mode=%s --points=%d --beginx=%s --endx=%s --domain=%s --domainx=%s --logx=%d >> %s.dat' % (dev, amp, omega, force, alpha, Dg, Dp, lmd, comp, block, paths, periods, spp, algorithm, trans, mode, points, beginx, endx, domain, domainx, logx, outb)
    output = open('%s.dat' % outb, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

#fig 6a, 6b

amp = 1.7754
omega = 0.1876
alpha = 0.77
Dg = 2.0e-05
domainx = 'g'
logx = 0
beginx = 0.7
endx = 0.9
for force in [0.025, 0.046]:
    if force == 0.025:
        outb = 'fig6a_bias'
    elif force == 0.046:
        outb = 'fig6b_bias'
    _cmd = './prog --dev=%d --amp=%s --omega=%s --force=%s --alpha=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --block=%d --paths=%d --periods=%s --spp=%d --algorithm=%s --trans=%s --mode=%s --points=%d --beginx=%s --endx=%s --domain=%s --domainx=%s --logx=%d >> %s.dat' % (dev, amp, omega, force, alpha, Dg, Dp, lmd, comp, block, paths, periods, spp, algorithm, trans, mode, points, beginx, endx, domain, domainx, logx, outb)
    output = open('%s.dat' % outb, 'w')
    print >>output, '#%s' % _cmd
    output.close()
    print _cmd
    cmd = commands.getoutput(_cmd)

#fig 6.8 (MJ MSC)

amp = 2.1614
omega = 0.2733
force = 0
alpha = 0.77
Dg = 2.0e-05
Dp = 0
lmd = 0
domainx = 'f'
beginx = 0
endx = 0.1

outb = 'fig6.8_bias' 
_cmd = './prog --dev=%d --amp=%s --omega=%s --force=%s --alpha=%s --Dg=%s --Dp=%s --lambda=%s --comp=%d --block=%d --paths=%d --periods=%s --spp=%d --algorithm=%s --trans=%s --mode=%s --points=%d --beginx=%s --endx=%s --domain=%s --domainx=%s --logx=%d >> %s.dat' % (dev, amp, omega, force, alpha, Dg, Dp, lmd, comp, block, paths, periods, spp, algorithm, trans, mode, points, beginx, endx, domain, domainx, logx, outb)
output = open('%s.dat' % outb, 'w')
print >>output, '#%s' % _cmd
output.close()
print _cmd
cmd = commands.getoutput(_cmd)

os.system('gnuplot test.plt')
os.system('mv -vf *.dat *.png %s' % DIRNAME)
