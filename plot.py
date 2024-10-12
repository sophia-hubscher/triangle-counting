import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

def get_timestamped_subfolder(parent_folder):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_subfolder = os.path.join(parent_folder, current_time)
    os.makedirs(new_subfolder, exist_ok=True)

    return new_subfolder

plots_folder = get_timestamped_subfolder('plots')

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
    plt.savefig(f'{plots_folder}/avg_error_{method}.png')

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
    plt.savefig(f'{plots_folder}/avg_duration_{method}.png')

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
    plt.savefig(f'{plots_folder}/avg_variance_{method}.png')

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
    plt.savefig(f'{plots_folder}/percent_error_{method}.png')

    for i, power in enumerate(powers):
        plt.figure(figsize=(12, 6))
        estimates_per_power = [results["all_estimates"][power][s] for s in s_values]
        plt.boxplot(estimates_per_power, positions=range(len(s_values)))
        plt.xticks(range(len(s_values)), s_values)
        plt.xlabel('Sample Size')
        plt.ylabel('Estimate Distribution')
        plt.title(f'Whisker Plot of Estimates vs. Sample Size for Method: {method}, Power {power}')
        plt.grid(True)
        plt.savefig(f'{plots_folder}/whisker_plot_{method}_power_{power}.png')

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

    plot_comparison(all_results, methods)

def plot_comparison(all_results, methods):
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    markers = ['o', 's', 'D', '^', 'v', '<', '>']

    plt.figure(figsize=(12, 6))

    for method_idx, method in enumerate(methods):
        color = colors[method_idx % len(markers)]
        
        for power_idx, power in enumerate(all_results[method]['avg_errors']):
            marker = markers[power_idx % len(colors)]
            avg_errors = []
            avg_times = []

            for s in sorted(all_results[method]['s_values']):
                avg_errors.append(all_results[method]['avg_errors'][power][s])
                avg_times.append(all_results[method]['avg_times'][power][s])

            # Plotting accuracy vs runtime
            plt.plot(avg_times, avg_errors, marker=marker, linestyle='-', color=color, label=f'{method} Power {power}')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Average Runtime (seconds, log scale)')
    plt.ylabel('Average Error (log scale)')
    plt.title('Accuracy vs. Runtime Comparison Across Methods')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/accuracy_vs_runtime_comparison.png')

if __name__ == '__main__':
    plot_csv('estimation_results.csv')
