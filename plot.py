import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

dataset = 'fb'
four_clique = False
dataset_names = {'fb': 'Facebook Dataset',
                 'croc': 'Crocodile Wikipedia Dataset',
                 'GrQc': 'Collaboration Network Dataset',
                 'ba': 'Synthetic Barabasi Albert Dataset',
                 'wsa': 'Synthetic Watts Strogatz Albert Dataset'}

n_values = {'fb': 4039,
                'croc': 11631,
                'GrQc': 5242,
                'ba': 4000,
                'wsa': 4000}

true_triangle_counts ={'fb': 1612010,
                'croc': 630879,
                'GrQc': 48296,
                'ba': 1678322,
                'wsa': 877830}

sum_of_squared_node_triangle_counts ={'fb': 35518682856,
                'croc': 8785005059,
                'GrQc': 82839893,
                'ba': 40244099146,
                'wsa': 1749193288}

def get_timestamped_subfolder(parent_folder):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_subfolder = os.path.join(parent_folder, f'{dataset}/{current_time}')
    os.makedirs(new_subfolder, exist_ok=True)

    return new_subfolder

parent_folder_name = 'plots/4-clique' if four_clique else 'plots'
plots_folder = get_timestamped_subfolder(parent_folder_name)

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
    plt.title(f'Average Error vs. Sample Size for Method: {method} ({dataset_names[dataset]})')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{plots_folder}/avg_error_{method}.png')
    plt.close()

    plt.figure(figsize=(12, 5))
    for i, power in enumerate(powers):
        avg_times = [results["avg_times"][power][s] for s in s_values]
        plt.plot(s_values, avg_times, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (log scale)')
    plt.ylabel('Average Duration (seconds, log scale)')
    plt.title(f'Average Duration vs. Sample Size for Method: {method} ({dataset_names[dataset]})')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{plots_folder}/avg_duration_{method}.png')
    plt.close()

    plt.figure(figsize=(12, 5))
    for i, power in enumerate(powers):
        avg_variances = [results["avg_variances"][power][s] for s in s_values]
        plt.plot(s_values, avg_variances, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (log scale)')
    plt.ylabel('Average Variance (log scale)')
    plt.title(f'Average Variance vs. Sample Size for Method: {method} ({dataset_names[dataset]})')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{plots_folder}/avg_variance_{method}.png')
    plt.close()

    plt.figure(figsize=(12, 5))
    for i, power in enumerate(powers):
        avg_percent_errors = [results["avg_percent_errors"][power][s] for s in s_values]
        plt.plot(s_values, avg_percent_errors, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (log scale)')
    plt.ylabel('Percent Error (log scale)')
    plt.title(f'Percent Error vs. Sample Size for Method: {method} ({dataset_names[dataset]})')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{plots_folder}/percent_error_{method}.png')
    plt.close()

    if method == 'Importance Sampling' and not four_clique: # only plots for power=1 (i.e. uniform sampling)
        n = n_values[dataset]
        true_triangle_count = true_triangle_counts[dataset]

        plt.figure(figsize=(12, 5))

        filtered_s_values = [s for s in s_values if s != 0]

        squared_node_triangle_counts_sum = np.sum(sum_of_squared_node_triangle_counts[dataset])
        estimated_squared_errors = [((n / (9 * s)) * np.sum(squared_node_triangle_counts_sum)) - (true_triangle_count ** 2 / s) for s in filtered_s_values]
        avg_variances = [results["avg_variances"][0][s] for s in filtered_s_values]

        plt.plot(avg_variances, estimated_squared_errors, marker='o', linestyle='-', color=colors[0])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Variance (log scale)')
        plt.ylabel('Estimated Variance (log scale)')
        plt.title(f'Actual Variance vs. Estimated Variance for Uniform Sampling')
        plt.grid(True)
        plt.savefig(f'{plots_folder}/avg_vs_estimated_variance_{method}.png')
        plt.close()

    for i, power in enumerate(powers):
        plt.figure(figsize=(12, 6))
        estimates_per_power = [results["all_estimates"][power][s] for s in s_values]
        plt.boxplot(estimates_per_power, positions=range(len(s_values)))
        plt.xticks(range(len(s_values)), s_values)
        plt.xlabel('Sample Size')
        plt.ylabel('Estimate Distribution')
        plt.title(f'Whisker Plot of Estimates vs. Sample Size for Method: {method} ({dataset_names[dataset]}), Power {power}')
        plt.grid(True)
        plt.savefig(f'{plots_folder}/whisker_plot_{method}_power_{power}.png')
        plt.close()

def plot_csv():
    methods = set()

    # This stores results for each method separately
    all_results = {}
    results_folder_name = 'results/4_clique' if four_clique else 'results'

    with open(f'{results_folder_name}/estimation_results_{dataset}.csv', mode='r') as file:
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
        plot(s_values, all_results[method], sorted(all_results[method]['avg_percent_errors'].keys()), method)

    plot_comparison(all_results, methods)
    plot_comparison(all_results, ["Variance Reduction", "Importance Sampling", "Variance Reduction + Importance Sampling"], True)

def plot_comparison(all_results, methods, only_power_2=False):
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    markers = ['o', 's', 'D', '^', 'v', '<', '>']

    plt.figure(figsize=(12, 6))

    for method_idx, method in enumerate(methods):
        color = colors[method_idx % len(markers)]
        
        for power_idx, power in enumerate(all_results[method]['avg_percent_errors']):
            if only_power_2:
                power = 2

            marker = markers[power_idx % len(colors)]
            avg_percent_errors = []
            avg_times = []

            for s in sorted(all_results[method]['s_values']):
                if (method == "Variance Reduction"):
                    power = 1
                
                avg_percent_errors.append(all_results[method]['avg_percent_errors'][power][s])
                avg_times.append(all_results[method]['avg_times'][power][s])

            # Plotting accuracy vs runtime
            method_label = f'{method}' if method == "Variance Reduction" else f'{method} Power {power}'
            plt.plot(avg_times, avg_percent_errors, marker=marker, linestyle='-', color=color, label=method_label)

            if only_power_2:
                break

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Average Runtime (seconds, log scale)')
    plt.ylabel('Average Percent Error (log scale)')
    plt.title(f'Percent Error vs. Runtime Comparison Across Methods ({dataset_names[dataset]})')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if only_power_2:
        plt.savefig(f'{plots_folder}/limited_method_percent_error_vs_runtime_comparison.png', bbox_inches='tight')
    else:
        plt.savefig(f'{plots_folder}/percent_error_vs_runtime_comparison.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))

    for method_idx, method in enumerate(methods):
        color = colors[method_idx % len(markers)]
        
        for power_idx, power in enumerate(all_results[method]['avg_percent_errors']):
            if only_power_2:
                power = 2

            marker = markers[power_idx % len(colors)]
            avg_percent_errors = []

            s_values = sorted(all_results[method]['s_values'])

            for s in s_values:
                if (method == "Variance Reduction"):
                    power = 1
                avg_percent_errors.append(all_results[method]['avg_percent_errors'][power][s])

            # Plotting percent error vs sample size
            method_label = f'{method}' if method == "Variance Reduction" else f'{method} Power {power}'
            plt.plot(s_values, avg_percent_errors, marker=marker, linestyle='-', color=color, label=method_label)

            if only_power_2:
                break

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sample Size (log scale)')
    plt.ylabel('Average Percent Error (log scale)')
    plt.title(f'Percent Error vs. Sample Size Comparison Across Methods ({dataset_names[dataset]})')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if only_power_2:
        plt.savefig(f'{plots_folder}/limited_method_percent_error_vs_sample_size_comparison.png', bbox_inches='tight')
    else:
        plt.savefig(f'{plots_folder}/percent_error_vs_sample_size_comparison.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_csv()
