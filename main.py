from ParameterOptimizer import *
import yfinance as yf

if __name__ == '__main__':

    # Choose strategy
    strategy = bt.strategies.MA_CrossOver

    # Get parameters to optimize
    params = {}
    for key, value in strategy.params.__dict__.items():
        if isinstance(value, (int, float)):  # Filter out non-optimizable parameters
            params[key] = value

    # Create constraints on parameters (Optional)
    constraints = {
        'fast': [lambda x: x > 0],
        'slow': [lambda x: x > 0],
    }

    # Choose one or more datasets
    print("Downloading data...")
    spy1 = yf.download('SPY', '2002-01-01', '2006-4-30')
    spy2 = yf.download('SPY', '2005-01-01', '2007-12-31')
    datas = [spy1, spy2]

    # Set weights of each dataset
    weights = [
        0.6, # 60% of evaluation is based on spy1
        0.4  # 40% of evaluation is based on spy2
    ]

    opt = ParameterOptimizer(
        strategy=strategy, 
        weights=weights,
        datas=datas, 
        cash=5000, # Starting cash for backtesting strategies
        generation_count=5, # Number on generations before stopping optimization
        population=10, # Number of parameter sets in each generation
        top_n=4 # Number of parameter sets to select from each generation
    )
    best_params = opt.optimize_parameters(params)
    opt.plot(best_params, spy1)
