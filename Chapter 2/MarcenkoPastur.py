import numpy as np, pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def mpPDF(var, q, pts):
    eMin, eMax = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5  # np.allclose(np.flip((eMax-eVal)), (eVal-eMin))==True
    pdf = pd.Series(pdf, index = eVal)
    return pdf
def getPCA(matrix):
    eVal, eVec = np.linalg.eig(matrix)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec
def fitKDE(obs, bWidth=.15, kernel='gaussian', x=None):
    if len(obs.shape) == 1: obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None: x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1: x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf
def main() :
    x = np.random.normal(size = (10000, 1000))
    eVal0, eVec0 = getPCA(np.corrcoef(x, rowvar = 0))
    pdf0 = mpPDF(1, q = x.shape[0] / float(x.shape[1]), pts = 1000)
    pdf1 = fitKDE(np.diag(eVal0), bWidth = 0.01)

    plt.figure(figsize = (6,4))
    plt.plot(pdf0, label='Marcenko-Pastur Distribution')
    plt.plot(pdf1, label='Empirical Distribution (KDE)')
    plt.xlabel(r'$\lambda$', fontfamily='Serif')
    plt.ylabel(r'prob($\lambda$)', fontfamily='Serif')
    plt.xlim([0, 2])
    plt.legend(loc='upper right', fontsize=9)
    plt.title('Marcenko-Pastur Distribution', fontsize=12, fontfamily='Serif')
    return None

if __name__ == '__main__' : main();