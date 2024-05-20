![GenTrader_Logo](https://github.com/jodhangill/GenTrader/assets/87293665/704c83f8-a6cf-4180-82fe-c33104f1d4d7)

# GenTrader
A genetic algorithm optimizer for optimizing trading strategies with the Backtrader library.

## Table of Contents

1. [About](#about)
2. [Getting Started](#getting-started)
3. [Adding a Strategy](#adding-a-strategy)
4. [Config Directory](#config-directory)
5. [Optimizer Configuration](#optimizer-configuration)
   - [Choosing a Strategy](#choosing-a-strategy)
   - [Setting Constraints](#setting-constraints)
   - [Choosing Stock Data](#choosing-stock-data)
   - [Configuring Genetic Algorithm](#configuring-genetic-algorithm)
     - [Tips](#tips-for-configuring-genetic-algorithm)
6. [Saving and Loading](#saving-and-loading)
   - [Example](#example)
7. [Output Preferences](#output-preferences)
8. [Evaluation](#evaluation)
   - [Arguments](#arguments)
9. [Acknowlegements](#acknowledgements)

## About

GenTrader is a powerful tool that leverages genetic algorithms in conjunction with the Backtrader Python library to optimize trading strategy parameters across any dataset. 

#### Why Use GenTrader?

When refining a trading algorithm, you might encounter multiple parameters needing optimization. 
For example, tuning three parameters within a range of 10 to 40 using a brute-force approach would require evaluating 30 x 30 x 30 = **27,000 backtests**. 
Such an exhaustive method can be extremely time-consuming, especially with large datasets.

GenTrader streamlines this process. By using genetic algorithms, 
GenTrader can significantly reduce the number of backtests required. For instance, running 5 generations with 10 parameter sets each results in only 5 x 10 = **50 backtests**. 
This efficiency allows for faster convergence to optimal parameters, saving both time and computational resources.

This makes GenTrader an invaluable tool for traders looking to optimize their strategies effectively and efficiently.

## Getting Started

1. Clone this repository:

```console
git clone https://github.com/jodhangill/GenTrader.git
```
    
2. Navigate to the project directory:

```console
cd ./GenTrader/
```
    
3. Install dependencies:
 
```console
pip install -r requirements.txt
```

4. Try it out by running the default configuration:

```console
python main.py
```



## Adding a Strategy

To add your own strategy, simply add a Python file with your Backtrader strategy into ```GenTrader/strategies/```.  

You can find a template file in ```GenTrader/strategies/template/```.  

To use your new strategy, you need to choose it in the [Configuration](#choosing-a-strategy).  

Tip: For more information on how to implement Backtrader strategies, see https://www.backtrader.com/docu/strategy/.  



## Config Directory

Everything you need to configure your optimization can be found in ```GenTrader/config/```:  

- ```evalutate.py``` contains the function used to evaluate the fitness of each parameter set after a backtest. See [Evaluation](#evaluation)

- ```initial_params.json``` stores the initial parameter set when loading non-default parameters. See [Saving and Loading](#saving-and-loading)

- ```optimizer_config.yaml``` contains various adjustable preferences for the optimization. See [Optimizer Configuration](#optimizer-configuration)  



## Optimizer Configuration

### Choosing a Strategy  

In ```config/optimizer_config.yaml```, choose a strategy from ```strategies/``` by entering its file name (excluding file extension) into ```name: ```.  

For example:  

To use the strategy from ```strategies/RsiThreshold.py```, you would enter:  

```yaml
name: RsiThreshold
```

---

### Setting Constraints

In ```config/optimizer_config.yaml```, set constraints on specific parameters by using the format ```- [PARAMETER_NAME] [OPERATOR] [VALUE]```  

For example:  

```yaml
constraints:
    - rsi_period > 0
    - rsi_low < 15
```

---

### Choosing Stock Data

In ```config/optimizer_config.yaml```, you can choose one or multiple past stock data to evaluate parameter sets on.  

Under ```stock_data``` you will find:  

- ```ticker```: Stock ticker. TIP: Lookup stock tickers at https://finance.yahoo.com/lookup/
- ```start_date```: The first date of the stock data
- ```end_date```: The ending date of the stock data
- ```weight```: The respective influence that each data set has on evaluation (percentage of influence = weight / sum of all weights)  
- ```starting_cash```: The amount of money the strategy will start with for each backtest  
- ```commission```: The commission rate of trades.  

To add stock data to the optimization, use the following format:  

```yaml
- ticker: [TICKER]
  start_date: YYYY-MM-DD
  end_date: YYYY-MM-DD
  weight: [FLOAT VALUE]
  starting_cash: [FLOAT VALUE]
  commission: [FLOAT VALUE]
```

For example:

```yaml
stock_data:
    - ticker: BTC-USD
      start_date: 2017-12-23
      end_date: 2019-01-23
      weight: 0.6
      starting_cash: 100000.0
      commission: 0.01

    - ticker: ETH-USD
      start_date: 2017-12-23
      end_date: 2019-01-23
      weight: 0.4
      starting_cash: 50000.0
      commission: 0.01
```

---

### Configuring Genetic Algorithm

Configuring the genetic algorithm is a very delicate process.  

Your decisions here will impact how efficiently and effectively GenTrader can optimize a specific strategy on the given datasets.

In ```config/optimizer_config.yaml```, you will find various values to adjust under ```genetic_algo_parameters```:  

- ```generation_count```: The number of generations computed before ending optimization (integer)
- ```population```: The number of parameter sets evaluated per generation (integer)
- ```selected```: The number of fittest parameter sets that are selected to be parents and used to produce next generation (integer)
- ```base_std```: The standard deviation of parameters mutations based on the normal distribution (float)
- ```relative_std```: Whether the standard deviation is scaled based on a parameter value's magnitude (boolean)
- ```crossover_rate```: The probability of a crossover occurring between two parents (float)
- ```seed```: Random number generation seed for reproducing runs (integer) (leave blank for random seed) 

#### Tips For Configuring Genetic Algorithm:

1. Keep in mind that the runtime is **O(G X P X D)** where:
   G = generation count
   P = population
   D = total number of days across all data sets

2. A good rule of thumb is that ```selected``` should be at least 10% of ```population```
3. Having a greater ```population``` than ```generation_count``` is typically more efficient
4. Generally, ```relative_std``` should be set to ```False``` when parameter values are close to 0
5. If the best fitness is not improving each generation, slightly increase ```base_std```
6. If the best fitness is still not improving after trying tip 4, increase ```population``` and ```selected```
7. If the best fitness is steadily improving during the entire run, increase ```generation_count``` or [rerun with last parameters](#example)

---

### Saving and Loading

You can choose whether to save or load in ```config/optimizer_config.yaml``` under ```saving_options```:  

```load_params```:  
- When set to ```False```, the initial population in generation 1 will be produced from the default parameter values of the strategy
- When set to ```True```, the initial population will be produced from the parameters in ```config/initial_parameters.json```

```save_params```:
- When set to ```False```, no data will be saved
- When set to ```True```:
    1. the best parameters/fitness will be saved to ```history/[date and time]/run_info.json```
    2. configuration will be saved to ```history/[date and time]/config/```
 
#### Example:
1. Use the default parameters for our initial population:  

    ```yaml
    saving_options: 
        load_params: False
        save_params:
    ```

2. Save the results of our run:    

    ```yaml
    saving_options: 
        load_params: False
        save_params: True
    ```

3. Run the optimizer:  

    ```console
    python main.py
    ```

4. Find the optimized parameters in ```history/[time of last run]/run_info.json```:  

    ```json
    {"best fitness": 0.035756989372899575, "best parameters": {"rsi_period": 14, "rsi_low": 31, "rsi_high": 57}}
    ```

5. Copy this parameter set to ```config/initial_params.json```:  
   ```json
   {"rsi_period": 14, "rsi_low": 31, "rsi_high": 57}
   ```

6. Load parameter from last run:  

    ```yaml
    saving_options: 
        load_params: True
        save_params: True
    ```

7. Rerun the optimizer:

    ```console
    python main.py
    ```

8. Check that the parameter loaded successfully:  

    ```console
    Base Parameter Set:
        rsi_period: 14
        rsi_low: 31
        rsi_high: 57
    ```

---

### Output Preferences

In ```config/optimizer_config.yaml```, you can change the format of the output under ```print_options:```

- ```max_list```: The maximum number of items listed for each parameter set (integer)
- ```detailed_progress```: True: logs progress at each generation, False: logs total progress percentage

---

## Evaluation

You can choose precisely how fitness is measured by implementing the evaluate function in ```config/evaluate.py```  

### Arguments  

1. ```sharpe_ratio```: Sharpe ratio of backtest
2. ```max_drawdown```: Maximum drawdown of backtest
3. ```total_compound_returns```: Total compound return of backtest
4. ```sqn```: System quality number of backtest
5. ```all_analyzers```: A collection holding various built-in analyzers  
    - Usage: ```all_analyzers.[ANALYZER NAME].get_analysis()```  
    - Analyzer names:  
      - [sharpe](https://www.backtrader.com/docu/analyzers-reference/#sharperatio)
      - [drawdown](https://www.backtrader.com/docu/analyzers-reference/#drawdown)
      - [returns](https://www.backtrader.com/docu/analyzers-reference/#returns)
      - [tradeanalyzer](https://www.backtrader.com/docu/analyzers-reference/#tradeanalyzer)
      - [sqn](https://www.backtrader.com/docu/analyzers-reference/#sqn)
      - [timereturn](https://www.backtrader.com/docu/analyzers-reference/#timereturn)
      - [annualreturn](https://www.backtrader.com/docu/analyzers-reference/#annualreturn)
      - [periodstats](https://www.backtrader.com/docu/analyzers-reference/#periodstats)
      - [sharperatio_a](https://www.backtrader.com/docu/analyzers-reference/#sharperatio_a)
      - [timedrawdown](https://www.backtrader.com/docu/analyzers-reference/#timedrawdown)
      - [grossleverage](https://www.backtrader.com/docu/analyzers-reference/#grossleverage)
      - [positionsvalue](https://www.backtrader.com/docu/analyzers-reference/#positionsvalue)
      - [pyfolio](https://www.backtrader.com/docu/analyzers-reference/#pyfolio)
      - [transactions](https://www.backtrader.com/docu/analyzers-reference/#transactions)

### Acknowledgements

GenTrader leverages the powerful [Backtrader](https://www.backtrader.com/) library for implementing and backtesting trading strategies. 
I greatly appreciate the efforts of the Backtrader community and contributors for providing such a comprehensive and flexible tool for trading strategy development.
