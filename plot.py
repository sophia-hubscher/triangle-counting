import numpy as np
import matplotlib.pyplot as plt
import csv

def plot(s_values, results, powers, method):
    colors = ['b', 'r', 'g', 'c', 'm', 'y']

    plt.figure(figsize=(12, 5))
    for i, power in enumerate(powers):
        avg_errors = [results["avg_errors"][power][s] for s in s_values]
        plt.plot(s_values, avg_errors, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (log scale)')
    plt.ylabel('Average Error (log scale)')
    plt.title(f'Average Error vs. Sample Size for Method: {method}')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    for i, power in enumerate(powers):
        avg_times = [results["avg_times"][power][s] for s in s_values]
        plt.plot(s_values, avg_times, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (log scale)')
    plt.ylabel('Average Duration (seconds, log scale)')
    plt.title(f'Average Duration vs. Sample Size for Method: {method}')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    for i, power in enumerate(powers):
        avg_variances = [results["avg_variances"][power][s] for s in s_values]
        plt.plot(s_values, avg_variances, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (log scale)')
    plt.ylabel('Average Variance (log scale)')
    plt.title(f'Average Variance vs. Sample Size for Method: {method}')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    for i, power in enumerate(powers):
        avg_percent_errors = [results["avg_percent_errors"][power][s] for s in s_values]
        plt.plot(s_values, avg_percent_errors, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (log scale)')
    plt.ylabel('Percent Error (log scale)')
    plt.title(f'Percent Error vs. Sample Size for Method: {method}')
    plt.grid(True)
    plt.legend()
    plt.show()

    for i, power in enumerate(powers):
        plt.figure(figsize=(12, 6))
        estimates_per_power = [results["all_estimates"][power][s] for s in s_values]
        plt.boxplot(estimates_per_power, positions=range(len(s_values)))
        plt.xticks(range(len(s_values)), s_values)
        plt.xlabel('Sample Size')
        plt.ylabel('Estimate Distribution')
        plt.title(f'Whisker Plot of Estimates vs. Sample Size for Method: {method}, Power {power}')
        plt.grid(True)
        plt.show()

def plot_csv(csv_file):
    methods = set()

    # This stores results for each method separately
    all_results = {}

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            method = row['Method']
            power = float(row['Power'])
            sample_size = int(row['Sample Size'])
            
            methods.add(method)

            if method not in all_results:
                all_results[method] = {
                    "avg_errors": {},
                    "avg_times": {},
                    "avg_variances": {},
                    "avg_percent_errors": {},
                    "all_estimates": {},
                    "s_values": set()
                }

            results = all_results[method]
            results['s_values'].add(sample_size)

            if power not in results['avg_errors']:
                results['avg_errors'][power] = {}
                results['avg_times'][power] = {}
                results['avg_variances'][power] = {}
                results['avg_percent_errors'][power] = {}
                results['all_estimates'][power] = {}

            results['avg_errors'][power][sample_size] = float(row['Avg Error'])
            results['avg_times'][power][sample_size] = float(row['Avg Time'])
            results['avg_variances'][power][sample_size] = float(row['Variance'])
            results['avg_percent_errors'][power][sample_size] = float(row['Avg Percent Error'])

            if sample_size not in results['all_estimates'][power]:
                results['all_estimates'][power][sample_size] = []

            estimates = list(map(float, row['Estimates'].split(',')))
            results['all_estimates'][power][sample_size].extend(estimates)

    for method in methods:
        s_values = sorted(all_results[method]['s_values'])
        plot(s_values, all_results[method], sorted(all_results[method]['avg_errors'].keys()), method)

if __name__ == '__main__':
    plot_csv('estimation_results.csv')
