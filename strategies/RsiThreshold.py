import backtrader as bt

class RsiThreshold(bt.Strategy):
    '''
    RSI Strategy:
        Buys when the RSI goes below the low threshold.
        Sells when the RSI goes above the high threshold.

    Parameters:
        rsi_period: Period for calculating the RSI (default: 14).
        rsi_low: Lower threshold for the RSI to trigger a buy signal (default: 30).
        rsi_high: Upper threshold for the RSI to trigger a sell signal (default: 70).
    '''
    params = (
        ('rsi_period', 14),
        ('rsi_low', 30),
        ('rsi_high', 70),
    )

    def __init__(self):
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data.close, period=self.params.rsi_period)

    def next(self):
        if self.rsi < self.params.rsi_low:
            self.buy()
        elif self.rsi > self.params.rsi_high:
            self.sell()
