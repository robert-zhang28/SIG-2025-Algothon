import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
import pred_model as pm

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

TIME_INTERVAL = 200
MIN_VOLATILITY = 0.01
CAPITAL = 10000 * 50

class Algorithm:
    """
    For all 3 strats - we can prob run a grid search to find optimal parameters
    """
    def __init__(self, filename):
        self.filename = filename
        self.nInst = 0
        self.nt = 0
        self.window = 30
        self.currentPos = None
        self.highly_corr_pairs = None
        self.coint_pairs = None
        self.stationary_pairs = None
        self.unused_instruments = None
        self.stationary_instruments = None
        self.paired_instruments = None
        self.prices = None
        self.quadratic_models = {}
        self.volatility = {}
        self.inverse_vol_weights = None

    def loadPrices(self):
        df=pd.read_csv(self.filename, sep='\s+', header=None, index_col=None)
        (nt,nInst) = df.shape
        self.nt = nt
        self.nInst = nInst
        self.currentPos = np.zeros(self.nInst)
        self.prices = (df.values).T
        return self.prices

    # prob just do correlation matrix and find highest correlations
    # need a bench mark correlation tho
    def find_pairs(self, threshold):
        """
        return all pairs that are highly correlated
        """
        corr_matrix = np.corrcoef(self.prices)
        n = corr_matrix.shape[0]
        high_corr_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                if corr_matrix[i, j] >= threshold:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
                    
        print(f"Found {len(high_corr_pairs)} highly correlated pairs (threshold={threshold}):")
        for i, j, corr in high_corr_pairs:
            print(f"Instrument {i} and {j} -> Corr = {corr:.3f}")
        self.highly_corr_pairs = high_corr_pairs

    def test_coint(self):
        coint_pairs = []
        for i, j, corr in self.highly_corr_pairs:
            _, p_value, _ = ts.coint(self.prices[i], self.prices[j])
            if p_value < 0.05:
                print(f"Instruments {i} and {j} are cointegrated (p-value = {p_value:.4f})")
                coint_pairs.append((i, j, corr))
        self.coint_pairs = coint_pairs

    def test_spread_stationarity(self):
        stationary_pairs = []

        for i, j, corr in self.coint_pairs:
            y = self.prices[i]
            x = self.prices[j]

            # OLS regression: y = alpha + beta * x
            x_with_const = sm.add_constant(x)
            model = sm.OLS(y, x_with_const).fit()
            alpha, beta = model.params

            # Compute spread
            spread = y - (alpha + beta * x)

            # ADF test
            result = adfuller(spread)
            p_value = result[1]

            if p_value < 0.05:
                print(f"Pair ({i}, {j}) has stationary spread (ADF p = {p_value:.4f})")
                stationary_pairs.append((i, j, corr))

        print(f"\nFound {len(stationary_pairs)} stationary pairs out of {len(self.coint_pairs)}")
        self.stationary_pairs = stationary_pairs

    #TODO: grid search to optimise parameters for arbitrage/ pairs trading strat
    def grid_search(self):
        pass
            
    def get_unused_instruments(self):
        used = set()
        for i, j, corr in self.stationary_pairs:
            used.add(i)
            used.add(j)
        n = self.prices.shape[0]
        print(n)
        unused = []
        for i in range(n):
            if i not in used:
                unused.append(i)
        return unused
        
    def test_stationary_instruments(self):
        unused_instruments = self.get_unused_instruments()
        stationary = []
        for i in unused_instruments:
            result = adfuller(self.prices[i])
            p_value = result[1]
            if p_value < 0.05:
                print(f"Instrument {i} ADF p-value: {p_value:.4f}")
                stationary.append(i)
        self.stationary_instruments = stationary
        
    def set_paired_instruments(self):
        result = set()
        for i, j, corr in self.stationary_pairs:
            result.add(i)
            result.add(j)
        paired_instruments = list(result)
        self.paired_instruments = paired_instruments
        
    def get_not_stationary_not_pairs_insts(self):
        not_stationary_not_pairs = []
        for i in range(self.nInst):
            if i not in self.paired_instruments and i not in self.stationary_instruments:
                not_stationary_not_pairs.append(i)
        return not_stationary_not_pairs
    
    def autocorr_lag1(self, prices):
        returns = np.diff(prices) / prices[:-1]  # daily returns
        if len(returns) < 2:
            return 0
        autocorr_values = acf(returns, nlags=1, fft=False)
        return autocorr_values[1]  # lag-1 autocorrelation
    
    def get_trend_instruments(self):
        not_stationary_not_pairs = self.get_not_stationary_not_pairs_insts()
        trend_instruments = []
        for i in not_stationary_not_pairs:
            prices = self.prices[i]
            autocorr = self.autocorr_lag1(prices)
            if autocorr > 0:
                trend_instruments.append(i)
        return trend_instruments
    
    def adx(self, high, low, close, period = 14):
        prev_close = close.shift(1)
        # True Range (TR)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        # Directional Movements
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=low.index)

        # Wilder's smoothing (EMA with alpha=1/period)
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return adx
    
    def compute_rsi(self, prices, window=14):
        delta = np.diff(prices)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)

        roll_up = pd.Series(up).rolling(window=window).mean()
        roll_down = pd.Series(down).rolling(window=window).mean()

        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    def calculate_volatility(self, prices, lookback=30):
        """Calculate annualized volatility based on daily returns"""
        if len(prices) < 2:
            return MIN_VOLATILITY
            
        returns = np.diff(prices) / prices[:-1]
        if len(returns) < lookback:
            return MIN_VOLATILITY
            
        recent_returns = returns[-lookback:]
        daily_vol = np.std(recent_returns)
        annualized_vol = daily_vol * np.sqrt(252)
        return max(annualized_vol, MIN_VOLATILITY)
    
    def compute_inverse_vol_weights(self, prcSoFar, lookback=30):
        """Compute normalized inverse volatility weights for all instruments."""
        weights = []
        for i in range(self.nInst):
            prices = prcSoFar[i]
            if len(prices) < lookback + 1:
                weights.append(0.0)
                continue
            returns = np.diff(prices[-lookback:]) / prices[-lookback:-1]
            vol = np.std(returns)
            inv_vol = 1.0 / max(vol, 1e-6)  # avoid div by 0
            weights.append(inv_vol)
        
        weights = np.array(weights)
        total = np.sum(weights)
        if total > 0:
            weights /= total
        return weights
        
    def update_volatilies(self, prcSoFar):
        for i in range(self.nInst):
            self.volatility[i] = self.calculate_volatility(prcSoFar[i])
        self.inverse_vol_weights = self.compute_inverse_vol_weights(prcSoFar)
        
    def train_models(self):
        for i in range(self.nInst):
            prices = self.prices[i]
            try:
                model = pm.QuadraticRegressionModel(degree=2, lookback=5)
                model.fit(prices)
                self.models[i] = model
            except Exception as e:
                print(f"Skipping instrument {i} due to error: {e}")
        
        
        
#NOTE: the sharpe ratios for the stocks being used on this strat are very bad
def trend_following(prcSoFar, long_window, short_window, adx_window=14, adx_threshold=30):
    trend_instruments = algo.get_trend_instruments()
    positions = np.zeros(prcSoFar.shape[0])
    
    for i in trend_instruments:
        prices = prcSoFar[i]
        if len(prices) < max(long_window, adx_window):
            continue
        
        price_series = pd.Series(prices)
        
        # Calculate short and long moving averages
        short_ma = price_series.rolling(window=short_window).mean()
        long_ma = price_series.rolling(window=long_window).mean()
        
        # Calculate ADX (requires high, low, close - approximate with close here for simplicity)
        # Use close for all three
        df = pd.DataFrame({'close': prices})
        df['high'] = 1.01 * df['close']
        df['low'] = 0.99 * df['close']
        
        adx_series = algo.adx(df['high'], df['low'], df['close'], period=adx_window)
        latest_adx = adx_series.iloc[-1]
        
        curr_short_ma = short_ma.iloc[-1]
        curr_long_ma = long_ma.iloc[-1]

        if np.isnan(curr_short_ma) or np.isnan(curr_long_ma) or np.isnan(latest_adx):
            continue
        
        max_pos = int(10000 // prices[-1])
        
        # Only trade if ADX indicates strong trend
        if latest_adx > adx_threshold:
            if curr_short_ma > curr_long_ma:
                positions[i] = max_pos
            elif curr_short_ma < curr_long_ma:
                positions[i] = -max_pos
            else:
                positions[i] = 0
        else:
            # ADX too low = no strong trend, close position
            positions[i] = 0
    return positions

#TODO: possibly find a better way of doing the signals other than just a simple threshold - maybe RSI?
def mean_reversion(stationary_instruments, prcSoFar, window, threshold):
    positions = np.zeros(prcSoFar.shape[0])
    for i in stationary_instruments:
        prices = prcSoFar[i]
        if len(prices) < window:
            continue
        rolling_mean = pd.Series(prices).rolling(window=window).mean()
        rolling_std = pd.Series(prices).rolling(window=window).std()
        if rolling_std.iloc[-1] == 0:
            continue
        z_score = (prices[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        
        max_pos = int(10000 // prices[-1])
        
        if z_score > threshold:
            # Price is above mean by threshold - short
            positions[i] = -max_pos
        elif z_score < -threshold:
            # Price below mean - long
            positions[i] = max_pos
        else:
            positions[i] = 0  # close position
    return positions

#TODO: start implementing getMyPosition
def getMyPosition(prcSoFar):
    (nins, nt) = prcSoFar.shape
    
    if algo.currentPos is None:
        algo.currentPos = np.zeros(prcSoFar.shape[0])
        
    if nt < algo.window:
        return algo.currentPos
    
    if nt % 7 == 0 or algo.inverse_vol_weights is None:
        algo.update_volatilies(prcSoFar)
    
    if algo.stationary_pairs is None or nt % TIME_INTERVAL == 0:
        algo.find_pairs(0.8)
        algo.test_coint()
        algo.test_spread_stationarity()
        algo.set_paired_instruments()
        algo.test_stationary_instruments()
        # res = algo.get_trend_instruments()
        # print(res)
        # print(len(res)) 
    
    weights = algo.inverse_vol_weights
    #print(weights)
    for i, j, corr in algo.stationary_pairs:
        inst1 = prcSoFar[i]
        inst2 = prcSoFar[j]
        price_i = prcSoFar[i, -1]
        price_j = prcSoFar[j, -1]
        
        inst2_const = sm.add_constant(inst2)
        model = sm.OLS(inst1, inst2_const).fit()
        alpha, beta = model.params

        # Calculate spread and z-score
        spread = inst1 - (alpha + beta * inst2)
        # mean_spread = np.mean(spread)
        # std_spread = np.std(spread)
        mean_spread = np.mean(spread)
        std_spread = np.std(spread)

        if std_spread == 0:
            continue  # Avoid division by zero

        z = (spread[-1] - mean_spread) / std_spread
        
        # Volatility-adjusted position sizing

        #TODO: figure out how to implement later not right now
        
        max_pos_inst1 = int(10000 // price_i)
        max_pos_inst2 = int(10000 // price_j)
        amount = min(max_pos_inst1, max_pos_inst2)
        
        if z > 1:
            algo.currentPos[i] -= amount
            algo.currentPos[j] += int(beta * amount)
        elif z < -1:
            algo.currentPos[i] += amount
            algo.currentPos[j] -= int(beta * amount)
        else:
            algo.currentPos[i] = 0
            algo.currentPos[j] = 0
    
    mean_rev_pos = mean_reversion(algo.stationary_instruments, prcSoFar, 60, 2.5)
    for i in algo.stationary_instruments:
        algo.currentPos[i] = mean_rev_pos[i]
        
    trend_following_pos = trend_following(prcSoFar, 60, 30)
    trend_instruments = algo.get_trend_instruments()
    for i in trend_instruments:
        algo.currentPos[i] = trend_following_pos[i]
        
    return algo.currentPos

def calcPL_per_instrument(prcHist, numTestDays):
    cash = 0
    curPos = np.zeros(algo.nInst)
    totDVolume = 0
    (_, nt) = prcHist.shape
    startDay = nt + 1 - numTestDays
    
    # Initialize daily PnL array per instrument
    dailyPL_per_inst = np.zeros((algo.nInst, numTestDays))
    
    value = 0
    
    for t in range(startDay, nt+1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]
        
        if (t < nt):
            newPosOrig = getMyPosition(prcHistSoFar)
            posLimits = np.array([int(x) for x in 10000 / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            
            comm = dvolume * 0.0005
            cash -= curPrices.dot(deltaPos) + comm
        else:
            newPos = np.array(curPos)
        
        # Calculate daily PnL per instrument:
        # PnL for each instrument = position[i] * (price[t] - price[t-1])
        if t > startDay:
            price_change = prcHist[:, t-1] - prcHist[:, t-2]
            dailyPL = curPos * price_change
            dailyPL_per_inst[:, t - startDay - 1] = dailyPL
        
        curPos = np.array(newPos)
        
        posValue = curPos.dot(curPrices)
        value = cash + posValue
    
    # Now compute metrics per instrument:
    mean_pl = np.mean(dailyPL_per_inst, axis=1)
    std_pl = np.std(dailyPL_per_inst, axis=1)
    sharpe = np.zeros(algo.nInst)
    
    # Annualize assuming 249 trading days
    for i in range(algo.nInst):
        if std_pl[i] > 0:
            sharpe[i] = np.sqrt(249) * mean_pl[i] / std_pl[i]
        else:
            sharpe[i] = 0
    return mean_pl, std_pl, sharpe, dailyPL_per_inst



pricesFile = "prices.txt"
algo = Algorithm(pricesFile)
prcAll = algo.loadPrices()
print ("Loaded %d instruments for %d days" % (algo.nInst, algo.nt))