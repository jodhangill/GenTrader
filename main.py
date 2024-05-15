from ParameterOptimizer import *
import yfinance as yf
import importlib
from datetime import datetime
import os
import json
import yaml

# Function to import strategy module dynamically
def import_strategy(strategy):
    """
    Import a trading strategy module dynamically.

    Args:
        strategy (str): Name of the strategy module.

    Returns:
        class: The imported trading strategy class.
    """
    module_query = 'strategies.' + strategy
    try:
        strategy_module = importlib.import_module(module_query)
        return getattr(strategy_module, strategy)
    except ModuleNotFoundError:
        print(f"Module '{strategy_module}' not found.")

# Function to parse constraint strings into lambda functions
def parse_constraints(constraint_strs):
    """
    Parse constraint strings into lambda functions.

    Args:
        constraint_strs (list): List of constraint strings.

    Returns:
        dict: Dictionary of parsed constraints.
    """
    constraints = {}
    for constraint in constraint_strs:
        # Split the condition string by whitespace
        parts = constraint.split()

        # Extract the parameter name and condition operator
        param, operator, value = parts

        # Create a lambda function based on the condition
        if operator == '>':
            condition_func = lambda x: x > float(value)
        elif operator == '<':
            condition_func = lambda x: x < float(value)
        elif operator == '>=':
            condition_func = lambda x: x >= float(value)
        elif operator == '<=':
            condition_func = lambda x: x <= float(value)
        elif operator == '==':
            condition_func = lambda x: x == float(value)
        else:
            raise ValueError("Invalid operator")

        # Store the lambda function for the parameter
        if param in constraints:
            constraints[param].append(condition_func)
        else:
            constraints[param] = [condition_func]
    return constraints

# Function to load stock data from Yahoo Finance
def load_stock_data(data_references):
    """
    Load stock data from Yahoo Finance.

    Args:
        data_references (list): List of dictionaries containing data references.

    Returns:
        tuple: A tuple containing lists of stock data and their respective weights.
    """
    print("Downloading data...")
    datas = []
    weights = []
    for data_ref in data_references:
        ticker = data_ref['ticker']
        start = data_ref['start_date']
        end = data_ref['end_date']
        weight = data_ref['weight']
        # Download stock data and append to lists
        data = yf.download(ticker, start, end)
        datas.append(data)
        weights.append(weight)
    return datas, weights

# Function to save optimization results to history
def save_to_history(params, cps, config):
    """
    Save optimization results to history.

    Args:
        params (dict): Optimized parameters.
        cps (float): Cumulative performance score.
        config (dict): Configuration settings.
    """
    current_datetime = datetime.now()
    folder_path = os.path.join("history/", str(current_datetime.date()) + ' ' + str(
        current_datetime.time()).replace(':', '·').replace('.', '·'))
    os.makedirs(folder_path)
    file_path = folder_path + "/best_params.json"

    # Save best parameters to JSON file
    with open(folder_path + "/best_params.json", 'w') as file:
        json.dump(params, file)
    
    # Save additional run info to JSON file
    run_info = {"cps": cps}
    with open(folder_path + "/run_info.json", 'w') as file:
        json.dump(run_info, file)

    # Save optimizer configuration to YAML file
    with open(folder_path + "/optimizer_config.yaml", 'w') as file:
        yaml.dump(config, file)

# Function to compare the structure of two dictionaries recursively
def compare_dicts_structure(dict1, dict2):
    """
    Compare the structure of two dictionaries recursively.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        bool: True if the dictionaries have the same structure, False otherwise.
    """
    # Base case: if both inputs are not dictionaries, return True if they are equal, otherwise False
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict1 == dict2

    # Check if the keys of dict1 are the same as the keys of dict2
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # Recursively compare the structures of the nested dictionaries
    for key in dict1.keys():
        if not compare_dicts_structure(dict1[key], dict2[key]):
            return False

    # If all checks passed, return True
    return True

# Main function to run the optimization process
def run():
    """
    Main function to run the optimization process.
    """
    # Load config file
    with open('optimizer_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set strategy
    strategy = import_strategy(config["strategy"]["name"])

    # Set parameters for initial population
    params = {}
    for key, value in strategy.params.__dict__.items(): # Get default parameters
        if isinstance(value, (int, float)):  # Filter out non-optimizable parameter types
            params[key] = value
    if (config["saving_options"]["load_params"]):
        with open("best_params.json", 'r') as file:
            loaded_params = json.load(file)
        # Check if loaded parameters are valid and replace parameters if valid
        if compare_dicts_structure(loaded_params, params):
            params = loaded_params
        else:
            print("Could not load last parameters: parameters are not valid for this strategy.")
    
    # Download all stock data
    datas, weights = load_stock_data(config["stock_data"])

    # Set parameter constraints
    constraints = parse_constraints(config["strategy"]["constraints"])

    # Parse the rest of the user's configs
    ga = config["genetic_algo_parameters"]
    generation_count = ga["generation_count"]
    population = ga["population"]
    top_n = ga["selected"]
    base_mutation_std = ga["bast_std"]
    relative_std = ga["relative_std"]
    crossover_rate = ga["crossover_rate"]
    
    seed = ga["seed"]
    if not isinstance(seed, (int, float, str, bytes, bytearray)):
        seed = None

    cash = config["trading"]["starting_cash"]
    commission = config["trading"]["commission"]

    print_options = config["print_options"]

    # Set optimizer configuration
    optimizer = ParameterOptimizer(
        strategy=strategy,
        datas=datas,
        weights=weights,
        constraints=constraints,
        generation_count=generation_count,
        population=population,
        top_n=top_n,
        base_mutation_std=base_mutation_std,
        relative_std=relative_std,
        crossover_rate=crossover_rate,
        seed=seed,
        cash=cash,
        commission=commission,
        print_options=print_options
    )

    # Perform optimization and get the best parameters and corresponding fitness
    best_params, cps = optimizer.optimize_parameters(params)

    # Save optimization results to history if specified in config
    if config["saving_options"]["save_params"]:
        save_to_history(best_params, cps, config)

    # Plot results if specified in config
    optimizer.plot(best_params, datas[0])

# Entry point of the script
if __name__ == '__main__':
    run()
