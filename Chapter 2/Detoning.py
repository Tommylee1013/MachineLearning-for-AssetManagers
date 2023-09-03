import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RandomMatrixWithSignal import cov2corr, getRndCov
from MarcenkoPastur import getPCA
from FittingMarcenkoPastur import findMaxEval
from ConstantResidual import denoisedCorr2, denoisedCorr

alpha, nCols, nFact, q = 0.995, 1000, 100, 10
def detoned_corr(corr, eigenvalues, eigenvectors, market_component=1):
    eigenvalues_mark = eigenvalues[:market_component, :market_component]
    eigenvectors_mark = eigenvectors[:, :market_component]
    corr_mark = np.dot(eigenvectors_mark, eigenvalues_mark).dot(eigenvectors_mark.T)
    corr = corr - corr_mark
    corr = cov2corr(corr)
    return corr

def main() :
    cov = np.cov(np.random.normal(size = (nCols * q, nCols)), rowvar = 0)
    cov = alpha * cov + (1-alpha) * getRndCov(nCols, nFact)
    corr0 = cov2corr(cov)
    eVal0, eVec0 = getPCA(corr0)

    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth = 0.01)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)

    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)
    eVal1, eVec1 = getPCA(corr1)

    denoised_eigenvalue = np.diag(eVal1)
    eigenvalue_prior = np.diag(eVal0)

    corr2 = detoned_corr(corr1, eVal1, eVec1, market_component = 1)
    eVal2, eVec2 = getPCA(corr2)
    detoned_eigenvalue = np.diag(eVal2)

    plt.figure(figsize = (6,4))
    plt.plot(eigenvalue_prior, label = 'Original eigen-function')
    plt.plot(denoised_eigenvalue, label = 'Denoised eigen-function', ls = '--')
    plt.plot(detoned_eigenvalue, label = 'Detoned-Denoised eigen-function', ls = ':')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$', fontfamily='Serif')
    plt.ylabel(r'number of $\lambda$', fontfamily='Serif')
    plt.legend(loc='upper right', fontsize=9)
    plt.title('Denoising with Constant residual eigenvalue method', fontsize=10, fontfamily='Serif')
    return None

if __name__ == '__main__' : main();