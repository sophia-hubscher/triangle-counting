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

def precompute_node_triangle_counts(A):
  node_triangle_counts = {}
  n = len(A)
  for node_index in range(n):
      node_triangle_counts[node_index] = count_node_triangles(A, node_index)
  return node_triangle_counts

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

def importance_estimate_per_node_method(A, s, power, node_triangle_counts):
  n = len(A)
  probabilities, sampled_nodes = sample_by_degree(A, n, s, power)

  estimate = 0
  
  for i in sampled_nodes:
    # triangle_count = count_node_triangles(A, i)
    triangle_count = node_triangle_counts[i]
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

"""# Plotting"""

def plot(s_values, results, powers):
  colors = ['b', 'r', 'g']

  # error graph
  plt.figure(figsize=(12, 5))
  for i, power in enumerate(powers):
      avg_errors = [results["avg_errors"][power][s] for s in s_values]
      plt.plot(s_values, avg_errors, marker='o', linestyle='-', color=colors[i], label=f'Power {power}')
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
      plt.plot(s_values, avg_times, marker='o', linestyle='-', color=colors[i], label=f'Power {power}')
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
      plt.plot(s_values, avg_variances, marker='o', linestyle='-', color=colors[i], label=f'Power {power}')
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
      plt.plot(s_values, avg_percent_errors, marker='o', linestyle='-', color=colors[i], label=f'Power {power}')
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

count_again = 0
def parallel_estimation(power, s, true_triangle_count, m, estimation_function, node_triangle_counts, iterations):
  errors = []
  times = []
  estimates = []
  percent_errors = []

  for _ in range(iterations):
    start_time = time.time()
    estimate = estimation_function(m, s, power, node_triangle_counts)
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

# run parallel estimation for multiple powers and sample sizes
def run_parallel_estimation(s_values, powers, true_triangle_count, m, estimation_function, iterations=20):
  node_triangle_counts = precompute_node_triangle_counts(m)

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
      parallel_results = pool.starmap(parallel_estimation, [(power, s, true_triangle_count, m, estimation_function, node_triangle_counts, iterations) for s in s_values])

      for i, result in enumerate(parallel_results):
        s = s_values[i]
        results["avg_errors"][power][s] = result["avg_error"]
        results["avg_times"][power][s] = result["avg_time"]
        results["avg_variances"][power][s] = result["variance"]
        results["avg_percent_errors"][power][s] = result["avg_percent_error"]
        results["all_estimates"][power][s] = result["estimates"]
        print(power, " ", s)

  return results

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

if __name__ == '__main__':
  file_path = 'facebook_combined.txt'
  node_count = 4039
  m = edges_to_adjacency_matrix_txt(file_path, node_count)

  s_values = [5, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, node_count]
  # s_values = [5, 100, 500, 1000, 2000, 3000, 4000, node_count]
  powers = [1, 1.5, 2]

  true_triangle_count = count_triangles(m)
  print(f"True Triangle Count: {true_triangle_count}")

  results = run_parallel_estimation(s_values, powers, true_triangle_count, m, importance_estimate_per_node_method)

  output_file = 'estimation_results.csv'
  method_name = 'importance_estimate_per_node_method'
  serialize_results_to_csv(results, s_values, powers, output_file, method_name)

  plot(s_values, results, powers)
  