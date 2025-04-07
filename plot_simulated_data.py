import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

dataset = 'ba'
dataset_names = {'fb': 'Facebook Dataset',
                 'croc': 'Crocodile Wikipedia Dataset',
                 'GrQc': 'Collaboration Network Dataset',
                 'ba': 'Synthetic Barabasi Albert Dataset'}

def get_timestamped_subfolder(parent_folder):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_subfolder = os.path.join(parent_folder, f'{dataset}/{current_time}')
    os.makedirs(new_subfolder, exist_ok=True)

    return new_subfolder

data = pd.read_csv(f"results/simulated/simulated_estimation_results_{dataset}.csv")
plots_folder = get_timestamped_subfolder('plots/simulated_plots')

methods = {
    "Multiplicative": (
        "Variance Reduction (Multiplicative Noise)",
        "Importance Sampling (Multiplicative Noise)",
    ),
    "Uniform": (
        "Variance Reduction (Uniform Noise)",
        "Importance Sampling (Uniform Noise)",
    ),
}

def plot_error_comparison_by_noise(noise_label, method1, method2):
    # Filter data for the selected methods
    noise_data = data[(data['Method'] == method1) | (data['Method'] == method2)]
    
    # Identify unique noise levels
    unique_noises = noise_data['Noise Scale'].unique()
    
    for noise_value in sorted(unique_noises):
        # Filter for the current noise level
        subset = noise_data[noise_data['Noise Scale'] == noise_value]
        
        # Group by method for plotting
        grouped = subset.groupby('Method')
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        
        for method_name, group in grouped:
            plt.plot(
                group["Sample Size"],
                group["Avg Error"],
                marker='o',
                label=method_name,
            )
        
        # Add titles and labels
        plt.title(f"Sample Size vs Avg Error for {noise_label} Noise (Scale={noise_value}) on {dataset_names[dataset]}")
        plt.xlabel("Sample Size")
        plt.ylabel("Avg Error")
        plt.legend()
        plt.grid()
        plt.savefig(f'{plots_folder}/percent_error_vs_sample_size_comparison_{noise_label.lower()}_{noise_value}.png', bbox_inches='tight')

# Generate plots for each noise type and level
for noise_label, (method1, method2) in methods.items():
    plot_error_comparison_by_noise(noise_label, method1, method2)