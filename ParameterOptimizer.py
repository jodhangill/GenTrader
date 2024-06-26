from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import backtrader as bt
from backtrader import TimeFrame
import heapq
import sys

# ParameterOptimizer optimizes given parameters using a genetic algorithm
class ParameterOptimizer:
    def __init__(self, strategy, weights, datas, cashes, commissions, evaluate, seed, base_mutation_std=0.1, relative_std=True, constraints=None, top_n=3, generation_count=5, population=10, crossover_rate=0.6, print_options={"max_list": -1, "detailed_progress": True}):
        """
        Initialize the ParameterOptimizer class.

        Args:
            strategy (class): The trading strategy class.
            weights (dict): Weights for performance on each dataset in datas.
            datas (list): List of data feeds to be used for backtest optimization.
            cashes (list): Starting cash values for each data backtest.
            commissions (list): Commission values for each data backtest.
            evaluate (func): Function of performance metrics used for evaluation.
            seed (number): Seed for random number generation
            base_mutation_std (float): Standard deviation for the mutation operation.
            relative_std (bool): Whether mutations use relative standard deviation.
            constraints (dict): Constraints for parameter values.
            top_n (int): Number of top parameter sets to select in each generation.
            generation_count (int): Number of generations for optimization.
            population (int): Population size for each generation.
            crossover_rate (float): Probability of crossover operation.
            print_options (dict): Formatting options for terminal outputs 
        """
        # Validate data and set random seed
        if len(weights) != len(datas):
            raise ValueError("Lengths of 'weights' and 'datas' must be equal")
        np.random.seed(seed)

        # Set class attributes
        self.strategy = strategy
        self.base_mutation_std = base_mutation_std
        self.relative_std = relative_std
        self.constraints = constraints
        self.commissions = commissions
        self.cashes = cashes
        self.top_n = top_n
        self.generation_count = generation_count
        self.population = population
        self.crossover_rate = crossover_rate
        self.weights = weights
        self.datas = datas
        self.print_options = print_options
        self.evaluate = evaluate

        self.completed_evaluations = 0
        self.total_evaluations = generation_count*population*len(datas)

    def init_parameter_sets(self, base_parameter_set):
        """
        Initialize parameter sets for the first generation.

        Args:
            base_parameter_set (dict): Base parameters for the strategy.

        Returns:
            list: List of parameter sets.
        """
        # Initialize with a base parameter set and create variations
        parameter_sets = [base_parameter_set]
        for _ in range(self.population - 1):
            parameter_sets.append(self.mutate_parameters(base_parameter_set))
        return parameter_sets

    def crossover(self, parent1, parent2):
        """
        Perform crossover operation between two parent parameter sets.

        Args:
            parent1 (dict): First parent parameter set.
            parent2 (dict): Second parent parameter set.

        Returns:
            tuple: Tuple containing two offspring parameter sets.
        """
        # Perform crossover with a certain probability
        if np.random.random() < self.crossover_rate:
            # Choose a random crossover point
            crossover_point = np.random.randint(0, len(parent1))

            # Create two offspring by combining the parents' parameters
            offspring1 = dict(list(parent1.items())[:crossover_point] + list(parent2.items())[crossover_point:])
            offspring2 = dict(list(parent2.items())[:crossover_point] + list(parent1.items())[crossover_point:])

            return offspring1, offspring2
        else:
            # If crossover doesn't occur, return the parents unchanged
            return parent1, parent2

    def mutate_parameters(self, parameters):
        """
        Mutate a set of parameters.

        Args:
            parameters (dict): Original set of parameters.

        Returns:
            dict: Mutated set of parameters.
        """
        # Mutate parameters with random values within constraints
        mutated_parameters = parameters.copy()
        for param_name, param_value in mutated_parameters.items():
            if self.relative_std:
                # Scale standard deviation
                mutation_std = self.base_mutation_std * abs(param_value)
            else:
                mutation_std = self.base_mutation_std

            # Generate new parameter value from normal distribution
            mutated_parameter = np.random.normal(loc=mutated_parameters[param_name], scale=mutation_std)

            if type(param_value) is int:
                mutated_parameter = int(np.round(mutated_parameter))
            if self.check_constraints(param_name, mutated_parameter):
                # Only mutate parameter if it satisfies constraints
                mutated_parameters[param_name] = mutated_parameter
        return mutated_parameters

    def check_constraints(self, parameter_name, parameter_value):
        """
        Check if a parameter satisfies all constraints.

        Args:
            parameter_name (str): Name of the parameter.
            parameter_value (float): Value of the parameter.

        Returns:
            bool: True if all constraints are satisfied, False otherwise.
        """
        # Check if the parameter value satisfies all constraints
        if self.constraints == None:
            return True
        if parameter_name not in self.constraints:
            return True
        for constraint in self.constraints[parameter_name]:
            if not constraint(parameter_value):
                return False
        return True

    def multiply(self, top_parameter_sets):
        """
        Perform crossover and mutation operations to generate new parameter sets.

        Args:
            top_parameter_sets (list): List of top parameter sets from previous generation.

        Returns:
            list: List of new parameter sets for the next generation.
        """
        # Perform crossover and mutation operations to create new parameter sets
        crossover_parameter_sets = top_parameter_sets.copy()

        i = 0
        j = 1

        # Perform crossover on different combinations of parents
        while len(crossover_parameter_sets) < self.population - 1:
            parent1 = top_parameter_sets[(i) % self.top_n]
            parent2 = top_parameter_sets[(i+j) % self.top_n]
            offspring1, offspring2 = self.crossover(parent1, parent2)
            crossover_parameter_sets.append(offspring1)
            crossover_parameter_sets.append(offspring2)
            i += 1
            if i == self.top_n/2:
                j += 1
                i = 0

        # If there is space, create a single offspring to bring population back up to size
        if len(crossover_parameter_sets) < self.population:
            parent1 = top_parameter_sets[(i) % self.top_n]
            parent2 = top_parameter_sets[(i+j) % self.top_n]
            offspring1, offspring2 = self.crossover(parent1, parent2)
            crossover_parameter_sets.append(offspring1)

        # Keep best parameter set of last generation unaltered
        mutated_parameter_sets = [top_parameter_sets[0]]

        for i in range(1, self.population):
            mutated_parameter_sets.append(self.mutate_parameters(crossover_parameter_sets[i]))

        return mutated_parameter_sets

    def evaluate_parameter_sets(self, parameter_sets):
        """
        Evaluate multiple sets of parameters and return the top n sets.

        Args:
            parameter_sets (list): List of parameter sets to evaluate.

        Returns:
            list: List of top parameter sets.
        """
        # Evaluate parameter sets based on backtesting metrics
        evaluated_parameters = []
        for params in parameter_sets:
            cps_values = []

            if self.print_options["detailed_progress"]:
                sys.stdout.write("Evaluating parameter set... (%d/%d completed) \r" % (len(evaluated_parameters), len(parameter_sets)))
                sys.stdout.flush()
            else:
                percentage_completed = (self.completed_evaluations / self.total_evaluations) * 100
                sys.stdout.write("Progress: %.2f%%    \r" % percentage_completed)
                sys.stdout.flush()

            cps_values = []  # Cumulative performance scores from each dataset

            # Compute CPS for each dataset
            for i, data in enumerate(self.datas):
                cerebro = bt.Cerebro()
                cerebro.addstrategy(self.strategy, **params)
                
                # Create a Data Feed
                cerebro.adddata(data)

                cerebro.broker.setcash(self.cashes[i])
                cerebro.broker.setcommission(self.commissions[i])

                # Add a FixedSize sizer according to the stake
                cerebro.addsizer(bt.sizers.SizerFix, stake=1)

                # Add analyzers for CPS
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
                cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tradeanalyzer")
                cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
                cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
                cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annualreturn")
                cerebro.addanalyzer(bt.analyzers.PeriodStats, _name="periodstats")
                cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name="sharperatio_a")
                cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name="timedrawdown")
                cerebro.addanalyzer(bt.analyzers.GrossLeverage, _name="grossleverage")
                cerebro.addanalyzer(bt.analyzers.PositionsValue, _name="positionsvalue")
                cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
                cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")

                # The following analyzers can sometimes throw unhandled math errors. 
                #cerebro.addanalyzer(bt.analyzers.VWR, _name="vwr")
                #cerebro.addanalyzer(bt.analyzers.LogReturnsRolling, _name="logreturnsrolling")
                #cerebro.addanalyzer(bt.analyzers.Calmar, _name="calmar")
                

                strats = cerebro.run()

                
                strat = strats[0]

                # Compile some basic metrics for easier accessibility when implementing evaluate()
                sharpe_ratio = strat.analyzers.sharpe.get_analysis()['sharperatio']
                max_drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
                total_compound_returns = strat.analyzers.returns.get_analysis()['rtot']
                sqn = strat.analyzers.sqn.get_analysis()
                basic_metrics = (
                    sharpe_ratio,
                    max_drawdown,
                    total_compound_returns,
                    sqn
                )

                # Calculate Cumulative performance score of this parameter set on the current dataset
                cps = self.evaluate(*basic_metrics, strat.analyzers)
                cps_values.append(cps)
                
                # Update stats
                self.completed_evaluations += 1

            # Sum CPS values according to their respective weights
            weighted_cps = 0
            for i in range(len(self.datas)):
                weighted_cps += cps_values[i]*self.weights[i]

            params = {}
            for key, value in strat.params.__dict__.items():
                if isinstance(value, (int, float)):  # Filter out non-optimizable parameters
                    params[key] = value

            evaluated_parameters.append((weighted_cps, params))

        if self.print_options["detailed_progress"]:
            # Clear progress line
            sys.stdout.write(" "*50 + "\r")
            sys.stdout.flush()

        # Return the top n tuples of (CPS value, parameter set)
        return evaluated_parameters

    def selection(self, parameter_sets):
        # Return top n parameter sets based on evaluation
        return heapq.nlargest(self.top_n, parameter_sets, key=lambda x: x[0])

    def optimize_parameters(self, base_parameter_set):
        """
        Optimize parameters using genetic algorithm.

        Args:
            base_parameter_set (dict): Base parameters for optimization.

        Returns: 
            tuple: (parameters, cps) Best parameter set achieved after all generations and its evaluation (CPS).
        """
        print("Base Parameter Set:")
        self.print_param_set(base_parameter_set, 1)
        parameter_sets = self.init_parameter_sets(base_parameter_set)
        
        print()
        print('\033[93m' + "Starting optimization..." + '\033[0m')
        print("_________________________________________________")

        # Perform genetic algorithm iterations
        for generation in range(self.generation_count):

            if self.print_options["detailed_progress"]:
                print()
                print('\033[1m' + '\033[4m' + '\033[95m' + f"Generation {generation + 1} of {self.generation_count}:" + '\033[0m')

            # Evaluate parameter sets and select top performers
            top_parameter_sets_with_cps = self.selection(
                self.evaluate_parameter_sets(parameter_sets))

            # Extract top parameter sets
            top_parameter_sets = [params for _, params in top_parameter_sets_with_cps]

            # Generate new parameter sets through crossover and mutation
            parameter_sets = self.multiply(top_parameter_sets)

            if self.print_options["detailed_progress"]:
                # Print generation statistics
                self.print_generation(top_parameter_sets_with_cps, generation)
        
        print('\033[1m' + '\033[93m' + "Done!            " + '\033[0m')
        return top_parameter_sets_with_cps[0][1], top_parameter_sets_with_cps[0][0]

    def plot(self, parameters, data):
        """
        Plot the strategy.

        Args:
            parameters (dict): Parameters for the strategy.
            data (pd.DataFrame): Data for the strategy.
        """
        cerebro = bt.Cerebro()
        cerebro.addstrategy(self.strategy, **parameters)
        cerebro.adddata(data)

        cerebro.broker.setcash(self.cashes[0])
        cerebro.broker.setcommission(self.commissions[0])

        cerebro.addsizer(bt.sizers.SizerFix, stake=1)
        cerebro.run()
        cerebro.plot()

    def print_param_set(self, param_set, indent):
        # Print parameter set with indentation
        print_count = self.print_options["max_list"]
        print_count
        for key, value in param_set.items():
            if print_count == 1:
                print(" "*(4*indent-1), f"{key}: {value}", end="")
                diff = len(param_set) - self.print_options["max_list"]
                if diff > 0:
                    print("...", f"({diff} more)")
                else:
                    print()
                break
            else:
                print(" "*(4*indent-1), f"{key}: {value}")
            print_count -= 1

    def print_generation(self, top_parameters_cps, i):
        # Print generation statistics
        print(" "*3, "Best evaluation: ", top_parameters_cps[0][0])
        print(" "*3, "Parameters:")
        self.print_param_set(top_parameters_cps[0][1], 2)