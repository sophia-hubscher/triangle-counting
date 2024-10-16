import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import csv
import time
import multiprocessing as mp
import os

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

def count_triangles(m):
  return np.trace(np.linalg.matrix_power(m, 3)) // 6.0

def gen_s_ints(s, n):
  choice_arr = sorted(random.choices(range(n), k=s))
  return choice_arr

def gen_sample_matrix(arr, A):
  return A[np.ix_(arr, arr)]

def count_node_triangles(A, node_index):
  neighbors = np.where(A[node_index] == 1)[0]
  triangle_count = 0

  # check each pair of neighbors to see if they form a triangle with the current node
  for i in range(len(neighbors)):
    for j in range(i + 1, len(neighbors)):
      if A[neighbors[i], neighbors[j]] == 1:
        triangle_count += 1

  return triangle_count

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

  triangle_count = 0
  for i in sampled_nodes:
    triangle_count += count_node_triangles(A, i)

  return triangle_count * (n / s) // 3

def estimate_uniformly_submatrix_method(A, s):
  n = len(A)
  sample_matrix = gen_sample_matrix(gen_s_ints(s, n), A)

  unscaled_tri_count = count_triangles(sample_matrix)
  scaled_tri_count = unscaled_tri_count * (n**3 / s**3)

  return scaled_tri_count

"""# Importance Sampling"""

def importance_estimate_per_node_method(A, s, power):
  n = len(A)
  probabilities, sampled_nodes = sample_by_degree(A, n, s, power)

  estimate = 0
  for i in sampled_nodes:
    triangle_count = count_node_triangles(A, i)
    estimate += triangle_count * (1 / (s * probabilities[i]))

  return estimate // 3

def importance_estimate_submatrix_method(A, s, power):
  n = len(A)
  probabilities, sampled_nodes = sample_by_degree(A, n, s, power)

  sampled_probabilities = [probabilities[i] for i in sampled_nodes]
  scaling_diagonal_values = 1 / (np.sqrt(np.multiply(sampled_probabilities, s)))
  scaling_matrix = np.diag(scaling_diagonal_values)

  unscaled_sample_matrix = gen_sample_matrix(sampled_nodes, A)
  scaled_sample_matrix = np.matmul(np.matmul(scaling_matrix, unscaled_sample_matrix), scaling_matrix)

  tri_count = count_triangles(scaled_sample_matrix)

  return tri_count

"""# Line of Best Fit """

def get_line_of_best_fit(A):
  n = len(A)
  degrees = np.sum(A, axis=1)
  triangles = np.zeros(n)

  for i in range(n):
    triangles[i] = count_node_triangles(A, i)

  valid_indices = (degrees > 0) & (triangles > 0)
  filtered_degrees = degrees[valid_indices]
  filtered_triangles = triangles[valid_indices]

  log_degrees = np.log(filtered_degrees)
  log_triangles = np.log(filtered_triangles)

  slope, intercept = np.polyfit(log_degrees, log_triangles, 1)

  # line_of_best_fit = slope * log_degrees + intercept
  # plt.scatter(log_degrees, log_triangles, color='blue', label='Data Points')
  # plt.plot(log_degrees, line_of_best_fit, color='red', label=f'Best Fit Line (slope={slope:.2f})')
  # plt.show()

  return slope, intercept

"""# Variance Reduction """

def estimate_variance_reduction_method(A, s, slope, intercept):
  n = len(A)

  degree_array = np.sum(A, axis=1)
  approx_triangles = np.power(degree_array, slope) * np.exp(intercept)
  M = np.sum(approx_triangles)

  sampled_nodes = gen_s_ints(s, n)
  sampled_node_triangles = np.array([count_node_triangles(A, i) for i in sampled_nodes])

  sampled_m_i_vals = np.array([approx_triangles[i] for i in sampled_nodes])

  D = np.sum(sampled_node_triangles - sampled_m_i_vals) * (n/s)

  # PLOTS FOR DEBUGGING

  # abs_diff = np.abs(sampled_m_i_vals - sampled_node_triangles)  # |m_i - delta_i|
  # relative_error = abs_diff / np.where(sampled_node_triangles != 0, sampled_node_triangles, 1)  # avoid division by 0

  # # true vs approximate triangle counts
  # plt.figure(figsize=(6, 4))
  # plt.scatter(sampled_node_triangles, sampled_m_i_vals, color='blue')
  # plt.plot([0, max(sampled_node_triangles)], [0, max(sampled_m_i_vals)], color='red', linestyle='--', label='y=x')
  # plt.xlabel('True Triangles (Δ_i)')
  # plt.ylabel('Approximate Triangles (m_i)')
  # plt.title('True vs Approximate Triangle Counts')
  # plt.legend()
  # plt.show()

  # # log-log plot of |m_i - delta_i| vs d_i
  # plt.figure(figsize=(6, 4))
  # plt.loglog(degree_array[sampled_nodes], abs_diff, 'o', color='green')
  # plt.xlabel('Degree (d_i)')
  # plt.ylabel('|m_i - Δ_i|')
  # plt.title('Log-Log Plot of |m_i - Δ_i| vs Degree (d_i)')
  # plt.show()

  # #  histogram of relative error |m_i - delta_i| / delta_i
  # plt.figure(figsize=(6, 4))
  # plt.hist(relative_error, bins=20, color='purple', edgecolor='black', alpha=0.7)
  # plt.xlabel('Relative Error |m_i - Δ_i| / Δ_i')
  # plt.ylabel('Frequency')
  # plt.title('Histogram of Relative Error')
  # plt.show()

  # # histogram of true triangles (Δ_i)
  # plt.figure(figsize=(6, 4))
  # plt.hist(sampled_node_triangles, bins=20, color='orange', edgecolor='black', alpha=0.7)
  # plt.xlabel('True Triangles (Δ_i)')
  # plt.ylabel('Frequency')
  # plt.title('Histogram of True Triangle Counts (Δ_i)')
  # plt.show()

  return (M + D) / 3

"""# Plotting"""

def plot(s_values, results, powers):
  colors = ['b', 'r', 'g', 'c', 'm', 'y']

  # error graph
  plt.figure(figsize=(12, 5))
  for i, power in enumerate(powers):
      avg_errors = [results["avg_errors"][power][s] for s in s_values]
      plt.plot(s_values, avg_errors, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Sample Size (log scale)')
  plt.ylabel('Average Error (log scale)')
  plt.title('Average Error vs. Sample Size for Different Powers')
  plt.grid(True)
  plt.legend()
  plt.show()

  # time graph
  plt.figure(figsize=(12, 5))
  for i, power in enumerate(powers):
      avg_times = [results["avg_times"][power][s] for s in s_values]
      plt.plot(s_values, avg_times, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Sample Size (log scale)')
  plt.ylabel('Average Duration (seconds, log scale)')
  plt.title('Average Duration vs. Sample Size for Different Powers')
  plt.grid(True)
  plt.legend()
  plt.show()

  # variance graph
  plt.figure(figsize=(12, 5))
  for i, power in enumerate(powers):
      avg_variances = [results["avg_variances"][power][s] for s in s_values]
      plt.plot(s_values, avg_variances, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Sample Size (log scale)')
  plt.ylabel('Average Variance (log scale)')
  plt.title('Average Variance vs. Sample Size for Different Powers')
  plt.grid(True)
  plt.legend()
  plt.show()

  # percent error graph
  plt.figure(figsize=(12, 5))
  for i, power in enumerate(powers):
      avg_percent_errors = [results["avg_percent_errors"][power][s] for s in s_values]
      plt.plot(s_values, avg_percent_errors, marker='o', linestyle='-', color=colors[i % len(colors)], label=f'Power {power}')
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Sample Size (log scale)')
  plt.ylabel('Percent Error (log scale)')
  plt.title('Percent Error vs. Sample Size for Different Powers')
  plt.grid(True)
  plt.legend()
  plt.show()

  # boxplot for estimates
  plt.figure(figsize=(12, 6))
  for i, power in enumerate(powers):
    estimates_per_power = [results["all_estimates"][power][s] for s in s_values]
    plt.boxplot(estimates_per_power, tick_labels=s_values)
    plt.xlabel('Sample Size')
    plt.ylabel('Estimate Distribution')
    plt.title(f'Whisker Plot of Estimates vs. Sample Size for Power {power}')
    plt.grid(True)
    plt.show()

"""# Test Sampling"""

def parallel_estimation(power, s, true_triangle_count, m, estimation_function, iterations):
  errors = []
  times = []
  estimates = []
  percent_errors = []

  for _ in range(iterations):
    start_time = time.time()
    estimate = estimation_function(m, s, power)
    end_time = time.time()

    duration = end_time - start_time
    error = abs(true_triangle_count - estimate)
    percent_error = abs(error / true_triangle_count)

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

def run_parallel_estimation(s_values, powers, true_triangle_count, m, estimation_function, iterations=20):
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
      parallel_results = pool.starmap(parallel_estimation, [(power, s, true_triangle_count, m, estimation_function, iterations) for s in s_values])

      for i, result in enumerate(parallel_results):
        s = s_values[i]
        results["avg_errors"][power][s] = result["avg_error"]
        results["avg_times"][power][s] = result["avg_time"]
        results["avg_variances"][power][s] = result["variance"]
        results["avg_percent_errors"][power][s] = result["avg_percent_error"]
        results["all_estimates"][power][s] = result["estimates"]
        print(power, " ", s)

  return results

def run_sequential_estimation(s_values, powers, true_triangle_count, m, estimation_function, iterations=20):
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
        error = abs(true_triangle_count - estimate)
        percent_error = abs(error / true_triangle_count)

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
  
  # file_path = 'data/ca-GrQc_mapped.txt'
  # node_count = 5242

  file_path = 'data/musae_crocodile.csv'
  node_count = 11631
  true_triangle_count = 630879

  m = edges_to_adjacency_matrix_csv(file_path, node_count)

  s_values = [500]
  # s_values = [5, 100, 500, 1000, 2000, 3000, 4000]
  powers = [0, 1, 1.5, get_line_of_best_fit(m)[0], 2]

  # true_triangle_count = count_triangles(m)
  print(f"True Triangle Count: {true_triangle_count}")

  # slope, intercept = get_line_of_best_fit(m)
  # estimate_variance_reduction_method(m, 500, slope, intercept)

  results = run_parallel_estimation(s_values, powers, true_triangle_count, m, importance_estimate_per_node_method)

  output_file = 'estimation_results_croc.csv'
  method_name = 'importance_estimate_per_node_method'
  serialize_results_to_csv(results, s_values, powers, f'results/{output_file}', method_name)

  plot(s_values, results, powers)
