import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RandomMatrixWithSignal import cov2corr, getRndCov
from MarcenkoPastur import getPCA
from FittingMarcenkoPastur import findMaxEval
from ConstantResidual import denoisedCorr2

def main() :
    alpha, nCols, nFact, q = 0.995, 1000, 100, 10

    cov = np.cov(np.random.normal(size = (nCols * q, nCols)), rowvar = 0)
    cov = alpha * cov + (1-alpha) * getRndCov(nCols, nFact)
    corr0 = cov2corr(cov)
    eVal0, eVec0 = getPCA(corr0)

    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth = 0.01)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)

    corr1 = denoisedCorr2(eVal0, eVec0, nFacts0, alpha = 0.5)
    eVal1, eVec1 = getPCA(corr1)

    denoised_eigenvalue = np.diag(eVal1)
    eigenvalue_prior = np.diag(eVal0)

    plt.figure(figsize = (6,4))
    plt.plot(eigenvalue_prior, label = 'Original eigen-function')
    plt.plot(denoised_eigenvalue, label = 'Denoised eigen-function', ls = '--')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$', fontfamily='Serif')
    plt.ylabel(r'number of $\lambda$', fontfamily='Serif')
    plt.legend(loc='upper right', fontsize=9)
    plt.title('Denoising with Targeted Shrinkage', fontsize=10, fontfamily='Serif')
    return None

if __name__ == '__main__' : main();