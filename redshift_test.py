#!/usr/bin/python

# Import the necessary libraries
import astropy.io.fits as pyfits
import gc
import numpy as np
import sys
import os
import time
from glob import glob

from astropy import constants as const, units as u
from astropy.coordinates import SkyCoord
from collections import Counter
from astropy.cosmology import LambdaCDM, z_at_value

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib import rc, rcParams

import modules_EG as utils


h, O_matter, O_lambda = [0.7, 0.25, 0.75]
cosmo = LambdaCDM(H0=h*100, Om0=O_matter, Ode0=O_lambda)

# angular_diameter_distance


zlens = 0.2
Dlens = cosmo.comoving_distance(zlens)/(1+zlens)
Dsat = (Dlens + 3*u.Mpc)

Dsource = 2 * Dlens

Sigma_crit_lens = Dsource / (Dlens * (Dsource-Dlens))
Sigma_crit_sat = Dsource / (Dsat * (Dsource-Dsat))

print(Sigma_crit_lens)
print(Sigma_crit_sat)
