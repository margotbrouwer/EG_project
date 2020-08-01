import scipy.stats as stats
import numpy as np

chisquared = 19.5496
dof = 15

print()
print('Chi^2=', chisquared)
print('DOF=', dof)

# Calculating the p-value from the chi-squared
pvalue = stats.chi2.cdf(chisquared, dof)

print()
print('P-value =', pvalue)


# Calculating the sigma- from the p-value
sigma = np.sqrt(stats.chi2.ppf(pvalue, 1.))
print('Sigma =', sigma)
print()
