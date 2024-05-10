from ParameterOptimizer import *
import yfinance as yf
from strategies import NeuralNetwork
import json

if __name__ == '__main__':

    # Choose strategy
    strategy = NeuralNetwork.NeuralNetwork

    # Load last best parameters to optimize
    file_path = "params.json"
    params = {}
    with open(file_path, 'r') as file:
        params = json.load(file)
    if not params:
        for key, value in strategy.params.__dict__.items():
            if isinstance(value, (int, float)):  # Filter out non-optimizable parameters
                params[key] = value

    # Create constraints on parameters (Optional)
    constraints = {
        # No constraints for neural net
    }

    # Choose one or more datasets
    print("Downloading data...")
    data1 = yf.download('SPY', '2007-12-01', '2024-01-30')
    #data2 = yf.download('BTC-USD', '2020-12-01', '2022-8-30')
    datas = [data1]

    # Set weights of each dataset
    weights = [
        0.6, # 60% of evaluation is based on data1
        #0.4  # 40% of evaluation is based on data2
    ]

    opt = ParameterOptimizer(
        strategy=strategy, 
        weights=weights,
        datas=datas, 
        cash=1000000, # Starting cash for backtesting strategies
        generation_count=5, # Number on generations before stopping optimization
        population=10, # Number of parameter sets in each generation
        top_n=1, # Number of parameter sets to select from each generation
        base_mutation_std=0.1
    )

    best_params = opt.optimize_parameters(params)

    # Save best parameter set (disabled for testing)
    #file_path = "params.json"
    #with open(file_path, "w") as file:
    #    json.dump(best_params, file)
    
    opt.plot(best_params, data1)
