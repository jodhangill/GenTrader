import backtrader as bt

class MovingAverageCrossover(bt.Strategy):
    """
    Moving Average Cross Strategy:
        Buys when the short-term moving average crosses above the long-term moving average.
        Sells when the short-term moving average crosses below the long-term moving average.

    Parameters:
        short_period: Period for the short-term moving average (default: 10).
        long_period: Period for the long-term moving average (default: 30).
    """
    params = (
        ('short_period', 10),
        ('long_period', 30),
    )

    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_period)
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_period)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        if self.crossover > 0:
            self.buy()
        elif self.crossover < 0:
            self.sell()
