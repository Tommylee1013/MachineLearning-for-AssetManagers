import numpy as np
from MarcenkoPastur import getPCA,mpPDF,fitKDE
import matplotlib.pyplot as plt

def getRndCov(nCols, nFacts):
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T)
    cov += np.diag(np.random.uniform(size=nCols))
    return cov
def cov2corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr
def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov

def main() :
    alpha, nCols, nFact, q = 0.995, 1000, 100, 10

    cov = np.cov(np.random.normal(size = (nCols * q, nCols)), rowvar = 0)
    cov = alpha * cov + (1-alpha) * getRndCov(nCols, nFact)
    corr0 = cov2corr(cov)
    eVal0, eVec0 = getPCA(corr0)
    x = np.random.normal(size=(10000, 1000))

    pdf0 = mpPDF(1, q = x.shape[0] / float(x.shape[1]), pts = 1000)
    pdf1 = fitKDE(np.diag(eVal0), bWidth = 0.01)

    plt.figure(figsize = (6,4))
    plt.grid(False)
    plt.plot(pdf0, label='Marcenko-Pastur Distribution', color = 'r', alpha = 0.6)
    plt.hist(np.diag(eVal0), density = True, bins = 500, label = "Empirical Distribution", color = 'orange', edgecolor = 'orange')
    plt.xlabel(r'$\lambda$', fontfamily='Serif')
    plt.ylabel(r'prob($\lambda$)', fontfamily='Serif')
    plt.legend(loc='upper right', fontsize=9)
    plt.title('Random Matrix with signal', fontsize=12, fontfamily='Serif')

if __name__ == '__main__' : main();