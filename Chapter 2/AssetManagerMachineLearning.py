import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pylab as plt
from scipy.optimize import minimize
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
import scipy.optimize as opt
from tqdm.notebook import tqdm


# snippet 2.1
# Marcenko-Pastur pdf
# q=T/N
def mpPDF(var, q, pts):
    eMin, eMax = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2  # calc lambda_minus, lambda_plus
    eVal = np.linspace(eMin, eMax, pts)  # Return evenly spaced numbers over a specified interval. eVal='lambda'
    # Note: 1.0/2*2 = 1.0 not 0.25=1.0/(2*2)
    pdf = q / (2 * np.pi * var * eVal) * (
                (eMax - eVal) * (eVal - eMin)) ** .5  # np.allclose(np.flip((eMax-eVal)), (eVal-eMin))==True
    pdf = pd.Series(pdf, index=eVal)
    return pdf


# snippet 2.2
# Test Marcenko-Pastur Thm
def getPCA(matrix):
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eig(matrix)  # complex Hermitian (conjugate symmetric) or a real symmetric matrix.
    indices = eVal.argsort()[::-1]  # arguments for sorting eval desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)  # identity matrix with eigenvalues as diagonal
    return eVal, eVec


def fitKDE(obs, bWidth=.15, kernel='gaussian', x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    # print(len(obs.shape) == 1)
    if len(obs.shape) == 1: obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    # print(x is None)
    if x is None: x = np.unique(obs).reshape(-1, 1)
    # print(len(x.shape))
    if len(x.shape) == 1: x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf


# snippet 2.3
def getRndCov(nCols, nFacts):  # nFacts - contains signal out of nCols
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T)  # random cov matrix, however not full rank
    cov += np.diag(np.random.uniform(size=nCols))  # full rank cov
    return cov


def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # for numerical errors
    return corr


def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov


# snippet 2.4 - fitting the marcenko-pastur pdf - find variance
# Fit error
def errPDFs(var, eVal, q, bWidth, pts=1000):
    var = var[0]
    pdf0 = mpPDF(var, q, pts)  # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    print("sse:" + str(sse))
    return sse


# find max random eVal by fitting Marcenko's dist
# and return variance
def findMaxEval(eVal, q, bWidth):
    out = minimize(lambda *x: errPDFs(*x), x0=np.array(0.5), args=(eVal, q, bWidth), bounds=((1E-5, 1 - 1E-5),))
    print("found errPDFs" + str(out['x'][0]))
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    eMax = var * (1 + (1. / q) ** .5) ** 2
    return eMax, var


# code snippet 2.5 - denoising by constant residual eigenvalue
# Remove noise from corr by fixing random eigenvalue
# Operation invariante to trace(Correlation)
# The Trace of a square matrix is the _Sum_ of its eigenvalues
# The Determinate of thematrix is the _Product_ of its eigenvalues
def denoisedCorr(eVal, eVec, nFacts):
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(
        eVal_.shape[0] - nFacts)  # all but 0..i values equals (1/N-i)sum(eVal_[i..N]))
    eVal_ = np.diag(eVal_)  # square matrix with eigenvalues as diagonal: eVal_.I
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)  # Eigendecomposition of a symmetric matrix: S = QΛQT
    corr1 = cov2corr(corr1)  # Rescaling the correlation matrix to have 1s on the main diagonal
    return corr1


# code snippet 2.6 - detoning
# ref: mlfinlab/portfolio_optimization/risk_estimators.py
# This method assumes a sorted set of eigenvalues and eigenvectors.
# The market component is the first eigenvector with highest eigenvalue.
# it returns singular correlation matrix:
# "the detoned correlation matrix is singualar, as a result of eliminating (at least) one eigenvector."
# Page 32
def detoned_corr(corr, eigenvalues, eigenvectors, market_component=1):
    """
    De-tones the de-noised correlation matrix by removing the market component.
    The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
    of the first eigenvalue that is above the maximum theoretical eigenvalue and the number of
    eigenvectors related to a market component.
    :param corr: (np.array) Correlation matrix to detone.
    :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
    :param eigenvectors: (float) Eigenvectors array.
    :param market_component: (int) Number of fist eigevectors related to a market component. (1 by default)
    :return: (np.array) De-toned correlation matrix.
    """

    # Getting the eigenvalues and eigenvectors related to market component
    eigenvalues_mark = eigenvalues[:market_component, :market_component]
    eigenvectors_mark = eigenvectors[:, :market_component]

    # Calculating the market component correlation
    corr_mark = np.dot(eigenvectors_mark, eigenvalues_mark).dot(eigenvectors_mark.T)

    # Removing the market component from the de-noised correlation matrix
    corr = corr - corr_mark

    # Rescaling the correlation matrix to have 1s on the main diagonal
    corr = cov2corr(corr)

    return corr


def test_detone():
    # ------ Test detone --------
    cov_matrix = np.array([[0.01, 0.002, -0.001],
                           [0.002, 0.04, -0.006],
                           [-0.001, -0.006, 0.01]])
    cor_test = np.corrcoef(cov_matrix, rowvar=0)
    eVal_test, eVec_test = getPCA(cor_test)
    eMax_test, var_test = findMaxEval(np.diag(eVal_test), q, bWidth=.01)
    nFacts_test = eVal_test.shape[0] - np.diag(eVal_test)[::-1].searchsorted(eMax_test)
    corr1_test = denoisedCorr(eVal_test, eVec_test, nFacts_test)
    eVal_denoised_test, eVec_denoised_test = getPCA(corr1_test)
    corr_detoned_denoised_test = detoned_corr(corr1_test, eVal_denoised_test, eVec_denoised_test)
    eVal_detoned_denoised_test, _ = getPCA(corr_detoned_denoised_test)
    np.diag(eVal_denoised_test)
    np.diag(eVal_detoned_denoised_test)

    expected_detoned_denoised_corr = np.array([1.56236229e+00, 1.43763771e+00, -2.22044605e-16])

    np.testing.assert_almost_equal(np.diag(eVal_detoned_denoised_test), expected_detoned_denoised_corr, decimal=4)
    np.testing.assert_almost_equal(sum(np.diag(eVal_denoised_test)), sum(np.diag(eVal_detoned_denoised_test)),
                                   decimal=4)
# import cvxpy as cp

# Code snippet 2.7
# Generate a block-diagnoal covariance matrix and a vector of means
def formBlockMatrix(nBlocks, bSize, bCorr):
    block = np.ones((bSize, bSize)) * bCorr
    block[range(bSize), range(bSize)] = 1  # diagonal is 1
    corr = block_diag(*([block] * nBlocks))
    return corr


def formTrueMatrix(nBlocks, bSize, bCorr):
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(.05, .2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0


def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov


# Code snippet 2.8
# generating the empirical covariance matrix
def simCovMu(mu0, cov0, nObs, shrink=False):
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=nObs)
    # print(x.shape)
    mu1 = x.mean(axis=0).reshape(-1, 1)  # calc mean of columns of rand matrix
    # print(mu1.shape)
    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=0)
    return mu1, cov1


# code snippet 2.9
# Denoising of the empirical covariance matrix
# by constant residual eigenvalue method
def deNoiseCov(cov0, q, bWidth):
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)  # denoising by constant residual eigenvalue method
    cov1 = corr2cov(corr1, np.diag(cov0) ** .5)
    return cov1


# code snippet 2.10
# Derive minimum-variance-portfolio
# Returns a column vector of percentage allocations
# should be subject to lagrangian constraints:
# 1. lambda_1*(sum(expectation(x_i)*x_i) - d = 0
# 2. lambda_2*(sum(x_i - 1))=0
# where d is expected rate of return
# w*=C^−1*μ/I.T*C^−1*μ - is minimum-variance-portfolio
# short sales are allowed
def optPort(cov, mu=None):
    inv = np.linalg.inv(
        cov)  # The precision matrix: contains information about the partial correlation between variables,
    #  the covariance between pairs i and j, conditioned on all other variables (https://www.mn.uio.no/math/english/research/projects/focustat/publications_2/shatthik_barua_master2017.pdf)
    ones = np.ones(shape=(inv.shape[0], 1))  # column vector 1's
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)  # def: w = w / sum(w) ~ w is column vector

    return w


# optPort with long only curtesy of Brady Preston
import cvxpy as cp
def optPort(cov,mu=None):
    n = cov.shape[0]
    if mu is None:mu = np.abs(np.random.randn(n, 1))
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov)
    ret =  mu.T @ w
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(risk),constraints)
    prob.solve(verbose=True)
    return np.array(w.value.flat).round(4)


# According to the question 'Tangent portfolio weights without short sales?'
# there is no analytical solution to the GMV problem with no short-sales constraints
# So - set the negative weights in WGV to 0, and make w sum up to 1
def optPortLongOnly(cov, mu=None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))  # column vector 1's
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)  # def: w = w / sum(w) ~ w is column vector
    w = w.flatten()
    threshold = w < 0
    wpluss = w.copy()
    wpluss[threshold] = 0
    wpluss = wpluss / np.sum(wpluss)

    return wpluss

def portfolio_annualised_performance(weights, day_returns, cov_matrix):
    returns = np.sum(day_returns.T.dot(weights)) *252
    std = np.sqrt(np.dot(weights, weights.dot(cov_matrix))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, day_returns, cov_matrix):
    '''
    Return performance of required number of random portfolios
    '''
    results = np.zeros((2,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        mu, sigma = 0.1, 0.1 # mean and standard deviation of the alternative vectors of expected returns
        weights = np.random.normal(mu, sigma, len(day_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, day_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
    return results, weights_record

def portfolio_volatility(weights, returns, cov_matrix):
    portfolio_returns = np.dot(returns.T, weights)
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    return portfolio_volatility

def efficient_return(day_returns, cov_matrix, target): # target is the target value for returning the efficient frontier axis (similiar to a y-axis value )
    num_assets = len(day_returns)
    args = (day_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, day_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = opt.minimize(portfolio_volatility, num_assets*[1./num_assets,],
                          args = args, method = 'SLSQP', bounds = bounds, constraints=constraints)
    return result

def efficient_frontier(day_returns, cov_matrix, returns_range): # return efficient frontier
    efficients = []
    for ret in tqdm(returns_range, desc = 'calculating efficient frontier using optimization method:'):
        efficients.append(efficient_return(day_returns, cov_matrix, ret))
    return efficients

def ef_with_random_portfolio_opt(day_returns,cov_matrix,num_portfolios,ax = None,return_plot = True):
    cov_matrix = day_returns.T.cov()
    results, weights = random_portfolios(num_portfolios,day_returns, cov_matrix)

    target = np.linspace(max(np.min(results[1]),0), np.quantile(results[1],0.7), 30)
    efficient_portfolios = efficient_frontier(day_returns, cov_matrix, target)
    frontier = [p['fun'] for p in efficient_portfolios]

    if return_plot:
        if not ax:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111)

            #ax.legend(labelspacing=0.8)
        #ax.plot(frontier, target, color = 'black', linewidth = 2, label = 'efficient frontier')

        ax.scatter(results[0,:],results[1,:], marker='o', s=10, alpha=0.3)
        ax.set_title('Calculated Portfolio Optimization based on Efficient Frontier')
        ax.set_xlabel('annualised volatility')
        ax.set_ylabel('annualised returns')
        #ax.legend()

    return ax,[frontier,target]

def efficient_return_simu(results, target):

    # return efficient frontier
    # use +-5% area of the target, return the nearest min
    results = pd.DataFrame(results.T).sort_values(by = 1)
    closiest_idx = np.argmin(np.abs(results[1]-target))
    data_target = results[1][closiest_idx]
    target_range_min = min(data_target*0.95, data_target*1.05)
    target_range_max = max(data_target*0.95, data_target*1.05)
    sub_results = results.loc[(results[1] <= target_range_max) & (results[1] >= target_range_min),0:2]

    return min(sub_results[0])

def efficient_frontier_emp(day_returns, cov_matrix, num_portfolios,  returns_range, random_seed=0):
    efficients = []
    np.random.seed(random_seed)
    results, weights = random_portfolios(
        num_portfolios, day_returns, cov_matrix)
    for ret in returns_range:
        efficients.append(efficient_return_simu(results, ret))
    return efficients

def ef_with_random_portfolio_simu(day_returns, cov_matrix, num_portfolios, ax=None, return_plot=True, random_seed=0,mean_frontier = None):
    results, weights = random_portfolios(
        num_portfolios, day_returns, cov_matrix)

    #target_start = max(results[1][results[0]==np.min(results[0])][0],0)
    target_start = 0
    if True: #not return_plot:
        target_end = 400
    else:
        target_end = results[1][results[0]==np.max(results[0])][0]

    target = np.linspace(target_start, target_end, 30)
    efficient_portfolios = None
    efficient_portfolios = efficient_frontier_emp(day_returns, cov_matrix, num_portfolios, target, random_seed=random_seed)

    if return_plot:
        if not ax:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111)

            # ax.legend(labelspacing=0.8)
        #if mean_frontier:
            #ax.plot(mean_frontier[0], mean_frontier[1], color='black',linewidth=2, label='mean efficient frontier')
        #else:
            #ax.plot(efficient_portfolios, target, color='black',linewidth=2, label='efficient frontier')

        ax.scatter(results[0, :], results[1, :], marker='o', s=10, alpha=0.3)
        ax.set_title(
            'Calculated Portfolio Optimization based on Efficient Frontier')
        ax.set_xlabel('annualised volatility')
        ax.set_ylabel('annualised returns')
        #ax.legend()
    else:
        ax = None

    return ax, [efficient_portfolios, target]

def MC_ef_frontier(day_returns, cov_matrix, itertimes = 100, random_seed = 42):
    np.random.seed(random_seed)
    target_ret = pd.DataFrame(day_returns.T)
    target_cov = cov_matrix
    frontier = []
    for i in tqdm(range(itertimes)):
        ax,frontier_ret = ef_with_random_portfolio_simu(target_ret,target_cov,1000,return_plot = False,random_seed = i)
        frontier.append(frontier_ret[0])
    mean_frontier_vol = np.mean(frontier, axis = 0)
    mean_frontier_ret = frontier_ret[1]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax,_ = ef_with_random_portfolio_simu(target_ret, target_cov, 1000, ax=ax,
                                         return_plot=True, random_seed = 42,
                                         mean_frontier = [mean_frontier_vol, frontier_ret[1]])
    return frontier

def error_mean_ef_frontier(frontiers, mean_frontier):
    err = []
    for frontier in frontiers:
        err.append(np.std(frontier-mean_frontier))
    return np.var(err)

def errPDFs(var, eVal, q, bWidth, pts = 1000):
    # Fit error
    var = var[0]
    pdf0 = mpPDF(var, q, pts) # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x = pdf0.index.values) # empirical pdf
    #import pdb; pdb.set_trace()
    sse=np.sum((pdf1-pdf0)**2)
    return sse

def random_eigenvalues(matrix):
    """
    Calculate the eigenvalues of the covariance matrix of the given matrix

    Parameters:
        - matrix: Input matrix

    Returns:
        - eigenvalues: Eigenvalues of the covariance matrix
    """
    covariance_matrix = np.cov(matrix.T)  # Calculate the covariance matrix
    eigenvalues = np.linalg.eigvals(covariance_matrix)  # Calculate the eigenvalues
    return eigenvalues