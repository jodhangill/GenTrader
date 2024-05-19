import backtrader as bt

class BollingerBandsStrategy(bt.Strategy):
    """
    Bollinger Bands Strategy:
        Buys when the price crosses below the lower Bollinger Band.
        Sells when the price crosses above the upper Bollinger Band.

    Parameters:
        period: Period for calculating the Bollinger Bands (default: 20).
        devfactor: Standard deviation factor for the Bollinger Bands (default: 2.0).
    """
    params = (
        ('period', 20),
        ('devfactor', 2.0),
    )

    def __init__(self):
        self.bbands = bt.indicators.BollingerBands(
            self.data.close, period=self.params.period, devfactor=self.params.devfactor)

    def next(self):
        if self.data.close < self.bbands.bot:
            self.buy()
        elif self.data.close > self.bbands.top:
            self.sell()
