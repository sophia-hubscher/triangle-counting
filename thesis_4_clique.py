import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import csv
import time
import multiprocessing as mp
import os
from itertools import combinations

"""# Matrix Generators"""

def gen_rand_matrix(n):
  upper_tri = np.triu(np.random.randint(0, 2, size=(n, n)), 1)

  symmetric_rand_matrix = upper_tri + upper_tri.T + np.diag(np.random.randint(0, 2, size=n))

  return symmetric_rand_matrix

def gen_zero_matrix(n):
  return np.zeros([n, n], dtype=int)

def gen_identity_matrix(n):
  return np.identity(n, dtype=int)

"""# Helpers"""

def edges_to_adjacency_matrix_csv(file_path, node_count):
  adjacency_matrix = np.zeros((node_count, node_count))

  with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)

    # skip the header
    next(csv_reader, None)

    for row in csv_reader:
      if len(row) >= 2:
        index_1 = int(row[0].strip())
        index_2 = int(row[1].strip())

        adjacency_matrix[index_1, index_2] = 1
        adjacency_matrix[index_2, index_1] = 1

  return adjacency_matrix

def edges_to_adjacency_matrix_txt(file_path, node_count):
  adjacency_matrix = np.zeros((node_count, node_count))

  with open(file_path, 'r') as file:
    for line in file:
      if line.strip():
        index_1, index_2 = map(int, line.split())

        adjacency_matrix[index_1, index_2] = 1
        adjacency_matrix[index_2, index_1] = 1

  return adjacency_matrix

def serialize_results_to_csv(results, s_values, powers, output_file, method_name):
  file_exists = os.path.isfile(output_file)

  with open(output_file, mode='a' if file_exists else 'w', newline='') as file:
    writer = csv.writer(file)

    if not file_exists:
      writer.writerow(["Method", "Power", "Sample Size", "Avg Error", "Avg Time", "Variance", "Avg Percent Error", "Estimates"])

    for power in powers:
      for s in s_values:
        avg_error = results["avg_errors"][power][s]
        avg_time = results["avg_times"][power][s]
        variance = results["avg_variances"][power][s]
        avg_percent_error = results["avg_percent_errors"][power][s]

        estimates_str = ', '.join(map(str, results["all_estimates"][power][s]))

        writer.writerow([method_name, power, s, avg_error, avg_time, variance, avg_percent_error, estimates_str])

def count_4_cliques(m):
    count = 0
    n = len(m)
    degrees = [sum(row) for row in m]
    
    for i in range(n):
        # loop through each pair of neighbors of node i
        for j in range(n):
            if m[i][j] == 1:
                for k in range(j + 1, n):
                    if m[i][k] == 1 and m[j][k] == 1:  # if j and k are neighbors of i and connected
                        # take the node with the lowest degree
                        triangle = sorted([i, j, k], key=lambda x: degrees[x])
                        lowest_degree_node = triangle[0]
                        
                        # iterate through the neighbors of the node with the lowest degree
                        for neighbor in range(k + 1, n):
                            if m[lowest_degree_node][neighbor] == 1:
                                # if the neighbor is connected to all nodes in the triangle, count a 4-clique
                                if all(m[neighbor][node] == 1 for node in triangle):
                                    count += 1
                                    break

        m[i] = [0] * n  # remove node i by setting its row to zero

    return count

def gen_s_ints(s, n):
  choice_arr = sorted(random.choices(range(n), k=s))
  return choice_arr

def gen_sample_matrix(arr, A):
  return A[np.ix_(arr, arr)]

def count_node_4_cliques(A, node_index):
    n = len(A)    
    clique_count = 0
    
    # get all nodes connected to node_index
    connected_nodes = [i for i in range(n) if A[node_index][i] == 1]
    
    # loop over all possible combinations of 3 other nodes from the connected ones
    for combo in combinations(connected_nodes, 3):
        i, j, k = combo

        # check if all pairs (i, j), (i, k), (j, k) are connected
        if A[i][j] == 1 and A[i][k] == 1 and A[j][k] == 1:
            clique_count += 1
                        
    return clique_count

def sample_by_degree(A, n, s, power):
  degrees = np.sum(A, axis=1)
  degrees_to_power = np.power(degrees, power)

  sum_of_degrees_to_power = np.sum(degrees_to_power)
  probabilities = degrees_to_power / sum_of_degrees_to_power

  sampled_nodes = random.choices(range(n), weights=probabilities, k=s)

  return probabilities, sampled_nodes

"""# Uniform Sampling"""

def estimate_uniformly_per_node_method(A, s):
  n = len(A)
  sampled_nodes = gen_s_ints(s, n)

  four_clique_count = 0
  for i in sampled_nodes:
    four_clique_count += count_node_4_cliques(A, i)

  return four_clique_count * (n / s) // 4

"""# Importance Sampling"""

def importance_estimate_per_node_method(A, s, power):
  n = len(A)
  probabilities, sampled_nodes = sample_by_degree(A, n, s, power)

  estimate = 0
  for i in sampled_nodes:
    four_clique_count = count_node_4_cliques(A, i)
    estimate += four_clique_count * (1 / (s * probabilities[i]))

  return estimate // 4

"""# Line of Best Fit """

def get_line_of_best_fit(A):
  n = len(A)
  degrees = np.sum(A, axis=1)
  four_cliques = np.zeros(n)
    
  for i in range(n):
    four_cliques[i] = count_node_4_cliques(A, i)

  valid_indices = (degrees > 0) & (four_cliques > 0)
  filtered_degrees = degrees[valid_indices]
  filtered_four_cliques = four_cliques[valid_indices]

  log_degrees = np.log(filtered_degrees)
  log_four_cliques = np.log(filtered_four_cliques)

  slope, intercept = np.polyfit(log_degrees, log_four_cliques, 1)

  # line_of_best_fit = slope * log_degrees + intercept
  # plt.scatter(log_degrees, log_four_cliques, color='blue', label='Data Points')
  # plt.plot(log_degrees, line_of_best_fit, color='red', label=f'Best Fit Line (slope={slope:.2f})')
  # plt.show()

  return slope, intercept

def get_line_of_best_fit_uniformly_sampled(A, s):
  n = len(A)
  sampled_nodes = gen_s_ints(s, n)
  
  degrees = np.array([sum(A[row]) for row in sampled_nodes])
  four_cliques = np.zeros(s)

  for counter, node in enumerate(sampled_nodes):
    four_cliques[counter] = count_node_4_cliques(A, node)

  valid_indices = (degrees > 0) & (four_cliques > 0)
  filtered_degrees = degrees[valid_indices]
  filtered_four_cliques = four_cliques[valid_indices]

  if (len(filtered_degrees) == 0):
    return 1, 1

  log_degrees = np.log(filtered_degrees)
  log_four_cliques = np.log(filtered_four_cliques)

  slope, intercept = np.polyfit(log_degrees, log_four_cliques, 1)

  return slope, intercept

"""# Variance Reduction """

def estimate_variance_reduction_method(A, s, power):
  n = len(A)

  slope, intercept = get_line_of_best_fit(A)

  degree_array = np.sum(A, axis=1)
  approx_four_cliques = np.power(degree_array, slope) * np.exp(intercept)
  M = np.sum(approx_four_cliques)

  sampled_nodes = gen_s_ints(s, n)
  sampled_node_four_cliques = np.array([count_node_4_cliques(A, i) for i in sampled_nodes])

  sampled_m_i_vals = np.array([approx_four_cliques[i] for i in sampled_nodes])

  D = np.sum(sampled_node_four_cliques - sampled_m_i_vals) * (n/s)

  return (M + D) / 4

def estimate_importance_variance_reduction_method(A, s, power):
  n = len(A)

  slope, intercept = get_line_of_best_fit(A)

  degree_array = np.sum(A, axis=1)
  approx_four_cliques = np.power(degree_array, slope) * np.exp(intercept)
  M = np.sum(approx_four_cliques)

  probabilities, sampled_nodes = sample_by_degree(A, n, s, power)
  sampled_node_probabilities = np.array([probabilities[i] for i in sampled_nodes])
  sampled_node_four_cliques = np.array([count_node_4_cliques(A, i) for i in sampled_nodes])

  sampled_m_i_vals = np.array([approx_four_cliques[i] for i in sampled_nodes])

  D = np.sum((sampled_node_four_cliques - sampled_m_i_vals) * (1 / (s * sampled_node_probabilities)))

  return (M + D) / 4

def estimate_sampled_line_importance_variance_reduction_method(A, s, power):
  n = len(A)

  slope, intercept = get_line_of_best_fit_uniformly_sampled(A, 50)

  degree_array = np.sum(A, axis=1)
  approx_four_cliques = np.power(degree_array, slope) * np.exp(intercept)
  M = np.sum(approx_four_cliques)

  probabilities, sampled_nodes = sample_by_degree(A, n, s, power)
  sampled_node_probabilities = np.array([probabilities[i] for i in sampled_nodes])
  sampled_node_four_cliques = np.array([count_node_4_cliques(A, i) for i in sampled_nodes])

  sampled_m_i_vals = np.array([approx_four_cliques[i] for i in sampled_nodes])

  D = np.sum((sampled_node_four_cliques - sampled_m_i_vals) * (1 / (s * sampled_node_probabilities)))

  return (M + D) / 4

"""# Test Sampling"""

def parallel_estimation(power, s, true_4_clique_count, m, estimation_function, iterations):
  errors = []
  times = []
  estimates = []
  percent_errors = []

  for _ in range(iterations):
    start_time = time.time()
    estimate = estimation_function(m, s, power)
    end_time = time.time()

    duration = end_time - start_time
    error = abs(true_4_clique_count - estimate)
    percent_error = abs(error / true_4_clique_count)

    errors.append(error)
    times.append(duration)
    estimates.append(estimate)
    percent_errors.append(percent_error)

  return {
    "avg_error": np.mean(errors),
    "avg_time": np.mean(times),
    "variance": np.var(estimates),
    "avg_percent_error": np.mean(percent_errors),
    "estimates": estimates
  }

def run_parallel_estimation(s_values, powers, true_4_clique_count, m, estimation_function, iterations=20):
  results = {
    "avg_errors": {power: {} for power in powers},
    "avg_times": {power: {} for power in powers},
    "avg_variances": {power: {} for power in powers},
    "avg_percent_errors": {power: {} for power in powers},
    "all_estimates": {power: {} for power in powers},
  }

  print("Method: ", estimation_function.__name__)

  with mp.Pool(mp.cpu_count()) as pool:
    for power in powers:
      parallel_results = pool.starmap(parallel_estimation, [(power, s, true_4_clique_count, m, estimation_function, iterations) for s in s_values])

      for i, result in enumerate(parallel_results):
        s = s_values[i]
        results["avg_errors"][power][s] = result["avg_error"]
        results["avg_times"][power][s] = result["avg_time"]
        results["avg_variances"][power][s] = result["variance"]
        results["avg_percent_errors"][power][s] = result["avg_percent_error"]
        results["all_estimates"][power][s] = result["estimates"]
        print(power, " ", s)

  return results

def run_sequential_estimation(s_values, powers, true_4_clique_count, m, estimation_function, iterations=20):
  results = {
    "avg_errors": {power: {} for power in powers},
    "avg_times": {power: {} for power in powers},
    "avg_variances": {power: {} for power in powers},
    "avg_percent_errors": {power: {} for power in powers},
    "all_estimates": {power: {} for power in powers},
  }

  for power in powers:
    for s in s_values:
      errors = []
      times = []
      estimates = []
      percent_errors = []

      for _ in range(iterations):
        start_time = time.time()
        estimate = estimation_function(m, s, power)
        end_time = time.time()

        duration = end_time - start_time
        error = abs(true_4_clique_count - estimate)
        percent_error = abs(error / true_4_clique_count)

        errors.append(error)
        times.append(duration)
        estimates.append(estimate)
        percent_errors.append(percent_error)

      results["avg_errors"][power][s] = np.mean(errors)
      results["avg_times"][power][s] = np.mean(times)
      results["avg_variances"][power][s] = np.var(estimates)
      results["avg_percent_errors"][power][s] = np.mean(percent_errors)
      results["all_estimates"][power][s] = estimates

      print(f"Power: {power}, s: {s}")

  return results

if __name__ == '__main__':
    # file_path = 'data/facebook_combined.txt'
    # node_count = 4039
    # true_4_clique_count = 1502405
  
    file_path = 'data/ca-GrQc_mapped.txt'
    node_count = 5242
    true_4_clique_count = 39101

    # file_path = 'data/musae_crocodile.csv'
    # node_count = 11631
    # true_4_clique_count = 451302

    # file_path = 'data/barabasi_albert.txt'
    # node_count = 4000
    # true_4_clique_count = 691107

    # file_path = 'data/watts_strogatz_albert.txt'
    # node_count = 4000
    # true_4_clique_count = 782479

    m = edges_to_adjacency_matrix_txt(file_path, node_count)

    slope, intercept = get_line_of_best_fit(m)
    degrees = np.sum(m, axis=1)
    
    s_values = [5, 100]
    # s_values = [5, 100, 500, 1000, 2000, 3000, 4000]
    powers = [0, 1, 1.5, get_line_of_best_fit(m)[0], 2]

    results = run_parallel_estimation(s_values, powers, true_4_clique_count, m, importance_estimate_per_node_method)

    output_file = 'estimation_results_croc.csv'
    method_name = 'importance_estimate_per_node_method'
    serialize_results_to_csv(results, s_values, powers, f'results/4_clique/{output_file}', method_name)
