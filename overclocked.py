'''Kelly-style leverage

Cross-asset clustering (mini market regimes)

MACD or other cross-confirmation filters'''

#****** BEST RN

'''import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    n, t = prcSoFar.shape

    if t < 252:
        return np.zeros(n)

    # === 1. Compute log returns ===
    returns = np.diff(np.log(prcSoFar), axis=1)

    # === 2. PCA to remove market component ===
    X = returns[:, -252:]
    X -= X.mean(axis=1, keepdims=True)
    cov = np.cov(X)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_pc = eigvecs[:, -1].reshape(-1, 1)
    market_component = top_pc @ (top_pc.T @ X)
    residuals = X - market_component

    # === 3. Momentum signal ===
    momentum = residuals[:, -20:].mean(axis=1)

    # === 4. EWMA Volatility (lambda = 0.94) ===
    lam = 0.94
    ewma_vol = np.sqrt(np.average((residuals[:, -60:] ** 2), axis=1, weights=np.power(lam, np.arange(60)[::-1])) + 1e-8)

    # === 5. Volume filter: ignore instruments with extreme volatility ===
    vol_thresh = np.percentile(ewma_vol, 90)
    mask = ewma_vol < vol_thresh

    momentum = momentum * mask
    ewma_vol = ewma_vol + (1 - mask) * 1e6  # Penalize blocked instruments

    # === 6. Rank by momentum ===
    N = 5
    ranked = np.argsort(momentum)
    short_idx = ranked[:N]
    long_idx = ranked[-N:]

    pos = np.zeros(n)
    capital = 1_000_000

    weights_long = capital / (2 * np.sum(1 / ewma_vol[long_idx]))
    for i in long_idx:
        pos[i] = weights_long / ewma_vol[i]

    weights_short = capital / (2 * np.sum(1 / ewma_vol[short_idx]))
    for i in short_idx:
        pos[i] = -weights_short / ewma_vol[i]

    pos = np.clip(np.round(pos / prcSoFar[:, -1]), -1000, 1000)
    currentPos = pos
    return currentPos
'''
#Result (best rn)
#mean(PL): 7.2
#return: 0.00013
#StdDev(PL): 339.85
#annSharpe(PL): 0.34 
#totDvolume: 10787369 
#Score: -26.75





'''import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    n, t = prcSoFar.shape

    if t < 252:
        return np.zeros(n)

    # === 1. Daily log returns ===
    log_returns = np.diff(np.log(prcSoFar), axis=1)

    # === 2. EWMA volatility ===
    vol_window = 20
    weights = np.exp(-np.arange(vol_window)[::-1] / 5)
    weights /= weights.sum()
    ewma_vol = np.sqrt((log_returns[:, -vol_window:] ** 2 @ weights)) + 1e-8

    # === 3. PCA residuals ===
    X = log_returns[:, -252:]
    X -= X.mean(axis=1, keepdims=True)
    cov = np.cov(X)
    eigvals, eigvecs = np.linalg.eigh(cov)
    market_component = eigvecs[:, -1].reshape(-1, 1)
    market_proj = market_component @ (market_component.T @ X)
    residuals = X - market_proj

    # === 4. Residual momentum ===
    momentum = residuals[:, -20:].mean(axis=1)

    # === 5. Z-score filtering for signal confidence ===
    z = (momentum - momentum.mean()) / (momentum.std() + 1e-8)
    long_idx = np.where(z > 1.0)[0]
    short_idx = np.where(z < -1.0)[0]

    # Cap to top/bottom 10
    long_idx = long_idx[np.argsort(-z[long_idx])][:10]
    short_idx = short_idx[np.argsort(z[short_idx])][:10]

    # === 6. Volatility-scaled risk allocation ===
    capital = 1_000_000
    pos = np.zeros(n)

    if len(long_idx) > 0:
        inv_vol_sum = np.sum(1 / ewma_vol[long_idx])
        risk_per_long = capital / 2 / inv_vol_sum
        for i in long_idx:
            pos[i] = risk_per_long / ewma_vol[i]

    if len(short_idx) > 0:
        inv_vol_sum = np.sum(1 / ewma_vol[short_idx])
        risk_per_short = capital / 2 / inv_vol_sum
        for i in short_idx:
            pos[i] = -risk_per_short / ewma_vol[i]

    # === 7. Convert to shares and cap ===
    pos = np.clip(np.round(pos / prcSoFar[:, -1]), -1000, 1000)
    currentPos = pos
    return currentPos'''


#Exp1 trying to improve the best score
#-- Results for this code
#mean(PL): 3.0
#return: 0.00004
#StdDev(PL): 433.79
#annSharpe(PL): 0.11 
#totDvolume: 11877646 
#Score: -40.41






















