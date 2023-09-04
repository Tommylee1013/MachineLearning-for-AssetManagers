import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from MarcenkoPastur import mpPDF, fitKDE
from RandomMatrixWithSignal import getPCA, getRndCov, cov2corr

alpha, nCols, nFact, q = 0.995, 1000, 100, 10

def errPDFs(var, eVal, q, bWidth, pts=1000):
    var = var[0]
    pdf0 = mpPDF(var, q, pts)  # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    #print("sse:" + str(sse))
    return sse

def findMaxEval(eVal, q, bWidth):
    out = minimize(lambda *x: errPDFs(*x), x0=np.array(0.5), args=(eVal, q, bWidth), bounds=((1E-5, 1 - 1E-5),))
    # print("found errPDFs" + str(out['x'][0]))
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    eMax = var * (1 + (1. / q) ** .5) ** 2
    return eMax, var

def main() :
    cov = np.cov(np.random.normal(size = (nCols * q, nCols)), rowvar = 0)
    cov = alpha * cov + (1-alpha) * getRndCov(nCols, nFact)
    corr0 = cov2corr(cov)
    eVal0, eVec0 = getPCA(corr0)

    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth = 0.01)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
    x = np.random.normal(size=(10000, 1000))
    pdf0 = mpPDF(var0, q = x.shape[0]/float(x.shape[1]), pts = 1000)
    pdf1 = fitKDE(np.diag(eVal0), bWidth = 0.01)

    plt.figure(figsize = (6,4))
    plt.grid(False)
    plt.plot(pdf0, label='Marcenko-Pastur Distribution', color = 'r', alpha = 0.6)
    plt.hist(np.diag(eVal0), density = True, bins = 500, label = "Empirical Distribution", color = 'orange')
    plt.xlabel(r'$\lambda$', fontfamily='Serif')
    plt.ylabel(r'prob($\lambda$)', fontfamily='Serif')
    plt.legend(loc='upper right', fontsize=9)
    plt.title('Fitting the Marcenko-Pastur distribution', fontsize=12, fontfamily='Serif')

if __name__ == '__main__' : main();