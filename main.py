from ParameterOptimizer import *
from config.evaluate import evaluate
import yfinance as yf
import yfinance.shared as shared
import importlib
from datetime import datetime
import os
import json
import yaml
import shutil
from colorama import init

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
        strategy_class = None
        for _, obj in strategy_module.__dict__.items():
            if isinstance(obj, type):
                strategy_class = obj
        return strategy_class
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
    if constraint_strs is None:
        return {}
    
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
    print('\033[93m' + "Downloading data..." + '\033[0m')
    datas = []
    weights = []
    cashes = []
    commissions = []
    for data_ref in data_references:
        ticker = data_ref['ticker']
        start = data_ref['start_date']
        end = data_ref['end_date']
        weight = data_ref['weight']
        cash = data_ref['starting_cash']
        commission = data_ref['commission']
        # Download stock data and append to lists
        data = yf.download(ticker, start, end)
        if shared._ERRORS:
            return None, None, None, None

        datas.append(data)
        weights.append(weight)
        cashes.append(cash)
        commissions.append(commission)
    return datas, weights, cashes, commissions

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
    os.makedirs(folder_path + "/config")

    # Copy initial params
    shutil.copy("config/initial_params.json", folder_path + "/config")
    
    # Save additional run info to JSON file
    run_info = {
        "best fitness": cps,
        "best parameters": params
    }
    with open(folder_path + "/run_info.json", 'w') as file:
        json.dump(run_info, file)

    # Save optimizer configuration to YAML file
    with open(folder_path + "/config/optimizer_config.yaml", 'w') as file:
        yaml.dump(config, file)

    # Copy evaluate.py
    shutil.copy("config/evaluate.py", folder_path + "/config")

# Function to compare the structure of two dictionaries
def compare_dicts_structure(dict1, dict2):
    """
    Compare the structure of two dictionaries.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        bool: True if the dictionaries have the same structure, False otherwise.
    """
    os.system('color')
    # Check if the keys of dict1 are the same as the keys of dict2
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # If all checks passed, return True
    return True

def print_logo():
    # Print coloured logo ascii art
    init(autoreset=True)
    ascii_art = [
        "                                                                                   ",
        " ██████╗ ███████╗███╗   ██╗████████ ██████   █████  ██████  ███████ ██████         ",
        "██╔════╝ ██╔════╝████╗  ██║   ██    ██   ██ ██   ██ ██   ██ ██      ██   ██       ╱",
        "██║  ███╗█████╗  ██╔██╗ ██║   ██    ██████  ███████ ██   ██ █████   ██████       ╱ ",
        "██║   ██║██╔══╝  ██║╚██╗██║   ██    ██   ██ ██   ██ ██   ██ ██      ██   ██   ╱╲╱  ",
        "╚██████╔╝███████╗██║ ╚████║   ██    ██   ██ ██   ██ ██████  ███████ ██   ██ _╱     ",
        " ╚═════╝ ╚══════╝╚═╝  ╚═══╝                                                ╱       ",
        "__________________________________________________________________________╱        ",
        "                                                                                   ",
    ]
    for line in ascii_art:
        print('\033[92m' + line[:27] + '\033[91m' + line[27:77] + '\033[92m' + line[77:79] + '\033[91m' + line[79:81] + '\033[92m' + line[81:83] + '\033[0m')


# Main function to run the optimization process
def run():
    """
    Main function to run the optimization process.
    """

    # Load config file
    with open('config/optimizer_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if config["print_options"]["display_logo"]:
        print_logo()

    # Set strategy
    strategy_name = config["strategy"]["name"]
    strategy = import_strategy(strategy_name)

    # Set parameters for initial population
    params = {}
    for key, value in strategy.params.__dict__.items(): # Get default parameters
        if isinstance(value, (int, float)):  # Filter out non-optimizable parameter types
            params[key] = value
    if (config["saving_options"]["load_params"]):
        with open("config/initial_params.json", 'r') as file:
            loaded_params = json.load(file)
        # Check if loaded parameters are valid and replace parameters if valid
        if compare_dicts_structure(loaded_params, params):
            params = loaded_params
        else:
            print("Could not load parameters: parameters are not valid for this strategy.")
    
    # Download all stock data
    datas, weights, cashes, commissions = load_stock_data(config["stock_data"])

    # Handle download fail
    if datas is None:
        print("Download failed!")
        print("Terminating run...")
        return

    # Set parameter constraints
    constraints = parse_constraints(config["strategy"]["constraints"])

    # Parse the rest of the user's configs
    ga = config["genetic_algo_parameters"]
    generation_count = ga["generation_count"]
    population = ga["population"]
    top_n = ga["selected"]
    base_mutation_std = ga["base_std"]
    relative_std = ga["relative_std"]
    crossover_rate = ga["crossover_rate"]
    
    seed = ga["seed"]
    if not isinstance(seed, int):
        # Generate seed based on current time
        seed = int(datetime.now().strftime("%Y%m%d%H%M%S")) % (2**32)

    print_options = config["print_options"]

    # Set optimizer configuration
    optimizer = ParameterOptimizer(
        evaluate=evaluate,
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
        cashes=cashes,
        commissions=commissions,
        print_options=print_options
    )

    print()
    print("Strategy: " + '\033[1m' + '\033[92m' + strategy_name + '\033[0m')
    print("Seed:", seed)

    # Perform optimization and get the best parameters and corresponding fitness
    best_params, cps = optimizer.optimize_parameters(params)

    # Save optimization results to history if specified in config
    if config["saving_options"]["save_params"]:
        save_to_history(best_params, cps, config)

    print('\033[93m' + "Plotting..." + '\033[0m')
    # Plot results if specified in config
    optimizer.plot(best_params, datas[0])

# Entry point of the script
if __name__ == '__main__':
    run()
