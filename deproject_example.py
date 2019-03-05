# example for testing
import numpy as np
from deproject.piecewise_powerlaw import esd_to_rho, _ESD
import matplotlib.pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages
r = np.logspace(np.log10(10), np.log10(1000), 10)
R = np.logspace(np.log10(10), np.log10(1000), 11)
rho = np.power(r, -2) / (2 * np.pi)
obs = np.power(r, -1) / np.pi
guess = np.power(r, -1.5) / np.pi
guess  = rho
extrapolate_outer = True
extrapolate_inner = True
inner_extrapolation_type = 'extrapolate'
startstep = np.min(-np.diff(np.log(guess))) / 3.  # probably reasonable
minstep = .001

best = esd_to_rho(
    obs,
    guess,
    r,
    R,
    extrapolate_inner=extrapolate_inner,
    extrapolate_outer=extrapolate_outer,
    inner_extrapolation_type=inner_extrapolation_type,
    startstep=.1,
    minstep=minstep,
    verbose=True

)

esd = _ESD(
    r,
    R,
    extrapolate_inner=extrapolate_inner,
    extrapolate_outer=extrapolate_outer,
    inner_extrapolation_type=inner_extrapolation_type
)
Rmids = np.power(10, .5 * (np.log10(R[:-1]) + np.log10(R[1:])))
with PdfPages('deproject.pdf') as pdffile:
    pp.figure(1)
    pp.xlabel(r'$\log_{10}r$')
    pp.ylabel(r'$\log_{10}\rho$')
    pp.plot(np.log10(r), np.log10(rho), '-b')
    pp.plot(np.log10(r), np.log10(guess), marker='o', mfc='None', mec='blue',
            ls='None')
    pp.savefig(pdffile, format='pdf')

    pp.figure(2)
    pp.xlabel(r'$\log_{10}R$')
    pp.ylabel(r'$\log_{10}\Delta\Sigma$')
    pp.plot(np.log10(Rmids), np.log10(esd(rho)), '-r')
    pp.plot(np.log10(Rmids), np.log10(esd(guess)), marker='o', mfc='None',
            mec='red', ls='None')
    pp.plot(np.log10(r), np.log10(exact), ':r')
    pp.savefig(pdffile, format='pdf')
