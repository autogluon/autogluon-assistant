import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
# Use a more universally available font
plt.rcParams['font.family'] = 'sans-serif'
# Use system defaults instead of specifying Arial
plt.rcParams['font.weight'] = 'bold'  # Set all fonts to bold by default

# Define the experimental settings
settings = ["Default", "No Summarization", "No Condensation"]
tasks = 8

# Success rates (1 or 0)
success_data = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],  # Default
    [1, 1, 1, 1, 1, 1, 1, 1],  # No Tutorial Summary
    [0, 1, 1, 1, 1, 1, 1, 1],  # No Tutorial Condensation
])

# Time used
time_data = np.array([
    [4089, 524, 370, 3699, 701, 2979, 800, 2066],  # Default (baseline)
    [2135, 614, 252, 6436, 738, 3173, 1269, 1897],  # No Tutorial Summary
    [4271, 527, 234, 2525, 701, 2896, 744, 1923],  # No Tutorial Condensation
])

# Token used
token_data = np.array([
    [79887, 30293, 54606, 72863, 29271, 199127, 14367, 25846],  # Default (baseline)
    [32538, 49683, 20105, 130637, 23970, 103502, 53964, 21610],  # No Tutorial Summary
    [94542, 35699, 28651, 92567, 34508, 185874, 20285, 26059],  # No Tutorial Condensation
])

# Calculate success rates
success_rates = np.mean(success_data, axis=1)

# Calculate relative times compared to default (baseline)
baseline_time = time_data[0]  # Default as baseline
relative_times = []

for i in range(len(settings)):
    relative_time_per_task = []
    for task in range(tasks):
        # Only include tasks where both current and baseline succeeded
        if success_data[i][task] > 0 and success_data[0][task] > 0 and time_data[i][task] > 0:
            relative_time_per_task.append(time_data[i][task] / baseline_time[task])
    
    # Calculate average of relative times for tasks that succeeded in both
    if relative_time_per_task:
        relative_times.append(np.mean(relative_time_per_task))
    else:
        relative_times.append(0)

# Calculate relative tokens compared to default (baseline)
baseline_token = token_data[0]  # Default as baseline
relative_tokens = []

for i in range(len(settings)):
    relative_token_per_task = []
    for task in range(tasks):
        # Only include tasks where both current and baseline succeeded
        if success_data[i][task] > 0 and success_data[0][task] > 0 and token_data[i][task] > 0:
            relative_token_per_task.append(token_data[i][task] / baseline_token[task])
    
    # Calculate average of relative tokens for tasks that succeeded in both
    if relative_token_per_task:
        relative_tokens.append(np.mean(relative_token_per_task))
    else:
        relative_tokens.append(0)

# Calculate time efficiency and token efficiency (reciprocals of relative metrics)
# For the baseline, these are just 1.0
time_efficiency = [1/rt if rt > 0 else 0 for rt in relative_times]
token_efficiency = [1/rt if rt > 0 else 0 for rt in relative_tokens]

# Print the calculated values for reference
print("Success Rates:", success_rates)
print("Relative Times:", relative_times)
print("Time Efficiency:", time_efficiency)
print("Relative Tokens:", relative_tokens)
print("Token Efficiency:", token_efficiency)

# Set up the plot - modified figsize to be wider and shorter (2/3 of original height)
fig, ax = plt.subplots(figsize=(16, 5.33), dpi=600)

# Positions for the bars
x = np.arange(len(settings))
width = 0.25  # the width of the bars

# Plotting the bars with edges
rects1 = ax.bar(x - width, success_rates, width, label='Success Rate', color='#F1FEC6',
                edgecolor='black', linewidth=1.5)  # Added edge to bars
rects2 = ax.bar(x, time_efficiency, width, label='Time Efficiency', color='#F39B6D',
                edgecolor='black', linewidth=1.5)  # Added edge to bars
rects3 = ax.bar(x + width, token_efficiency, width, label='Token Efficiency', color='#7BB2D9',
                edgecolor='black', linewidth=1.5)  # Added edge to bars

# Add horizontal line at y=1 for baseline reference
ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline (Default)')

# Set y-axis limit from 0.5 to 1.5
ax.set_ylim(0.5, 1.5)

# Add labels, title and legend with larger fonts and bold
ax.set_xlabel('Experimental Setting', fontsize=24, fontweight='bold', labelpad=10)  # Increased from 20
ax.set_ylabel('Performance Metrics', fontsize=24, fontweight='bold', labelpad=10)  # Increased from 20
ax.set_title('Performance Metrics Across Different Offline Indexing Settings', fontsize=28, fontweight='bold')  # Increased from 24
ax.set_xticks(x)
ax.set_xticklabels(settings, fontsize=22, fontweight='bold')  # Increased from 18
ax.tick_params(axis='y', labelsize=22)  # Increased from 18

# Make legend text larger and bold - arranged in 2x2 grid with edge color
ax.legend(loc='upper right', fontsize=18, frameon=True, ncol=4,
          framealpha=0.9, edgecolor='black', borderpad=0.8)  # Added edge color and increased font size

# Add grid
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Add value annotations with larger bold font
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=20, fontweight='bold')  # Increased from 16

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# Add border to the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_color('black')

plt.tight_layout()
# No need for subplots_adjust since legend is now inside the plot

plt.savefig("tutorial_sum_plots.pdf")
