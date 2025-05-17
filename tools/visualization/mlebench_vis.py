import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global font to be bold
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 14  # Increase base font size

# Function to create a radar chart
def radar_chart(fig, data, categories, groups, colors, title):
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax = fig.add_subplot(111, polar=True)
    
    # Draw diagonal lines (add these lines first so they appear behind data)
    for i in range(N):
        # Draw diagonal lines from center to each axis
        ax.plot([angles[i], angles[i]], [0, 1], color='#DDDDDD', linestyle='--', alpha=0.5, linewidth=4)
    
    # Improve label positioning
    label_angles = angles[:-1]  # Drop the last angle which is a repeat
    labels = []
    for i, cat in enumerate(categories):
        angle_deg = np.rad2deg(label_angles[i])
        # Adjust text alignment based on position
        if angle_deg == 0:
            ha, va = "center", "center"
        elif 0 < angle_deg < 180:
            ha, va = "center", "center"
        elif angle_deg == 180:
            ha, va = "center", "center"
        else:
            ha, va = "center", "center"
        
        # Set label closer to the graph (changed from 1.2 to 1.1)
        ax.text(label_angles[i], 1.1, cat, 
                size=32, color='#333333',  # Increased font size
                weight='bold',  # Make text bold
                horizontalalignment=ha, 
                verticalalignment=va)
    
    # Remove default xticks
    plt.xticks([])
    
    # Draw ylabels (removed for cleaner look)
    ax.set_rlabel_position(0)
    plt.yticks([], [])
    plt.ylim(0, 1)  # This will be overridden by individual scales
    
    # Compute max values for each category for normalization
    max_values = data.max(axis=1)
    min_values = data.min(axis=1)
    
    # Plot data
    for i, group in enumerate(groups):
        # Get the values and normalize them
        raw_values = data[group].values.flatten().tolist()
        # Normalize each value by its category's maximum (with a small buffer)
        values = []
        for j, val in enumerate(raw_values):
            # Special case for Rank: invert the normalization
            if categories[j] == 'Rank':
                # For Rank, lower is better, so invert the normalization
                best_rank = min_values.iloc[j]
                max_v = max_values.iloc[j] * 1.18
                # Higher normalized value means better (closer to outer edge)
                values.append((max_v - val) / ((max_v - best_rank) * 1.18))
            elif categories[j] == 'Success':
                # Start from 0.5
                values.append((val - 0.5) * 2 / ((max_values.iloc[j] - 0.5) * 2 * 1.18))
            else:
                # For other metrics, higher is better
                values.append(val / (max_values.iloc[j] * 1.18))
        
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=3, linestyle='solid', color=colors[i], label=group)  # Increased linewidth
        ax.fill(angles, values, color=colors[i], alpha=0.25)
    
    # We'll handle the legend in the main code instead
    # Not adding a legend here to avoid duplicates
    
    plt.title(title, size=36, color="#333333", y=1.05, weight='bold')  # Larger, bold title
    
    # Remove the circular grid and spines
    ax.grid(color='#DDDDDD', linestyle='--', alpha=0.7, linewidth=1.5)  # Thicker grid lines
    ax.spines['polar'].set_visible(False)
    
    return ax

# Create some random data (better looking random numbers for visualization)
np.random.seed(42)  # For reproducibility

# Create a DataFrame
categories = ['Rank', '#Gold', '#Silver+', '#Bronze+', '#Median+', 'Success']
groups = ['Ours', 'AIDE', 'MLAB', 'OpenHands']

# Create somewhat balanced data that looks meaningful
data = pd.DataFrame({
    'Ours': [1.43, 6, 8, 8, 12, 0.86],
    'AIDE': [2.36, 3, 3, 4, 8, 0.81],
    'MLAB': [3.29, 1, 1, 2, 2, 0.62],
    'OpenHands': [2.93, 2, 2, 3, 4, 0.71]
}, index=categories)

# Set up the figure with a white background and proper size
fig = plt.figure(figsize=(14, 12), facecolor="white")  # Changed background to white

# Define a pleasing color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create the radar chart
ax = radar_chart(fig, data, categories, groups, colors, "MLEBench-Lite Results")

# Add the actual values as text annotations for each axis with larger, bold font
for i, category in enumerate(categories):
    angle = i / float(len(categories)) * 2 * np.pi
    # Display the range for each category (moved closer to graph)
    max_val = data.loc[category].max() 
    min_val = data.loc[category].min() 
    
    # For Rank, show that lower values are better
    if category == 'Rank':
        range_text = f'[4-{min_val:.1f}] (lower is better)'
    elif category == 'Success':
        range_text = f'[0.5-{max_val:.1f}]'
    else:
        range_text = f'[0-{max_val:.1f}]'
        
    ax.text(angle, 0.6, range_text,  # Changed from 0.5 to 0.6 to move closer to edge
            horizontalalignment='center',
            verticalalignment='center',
            size=20, color='#555555',
            weight='bold')

# Create a truly larger legend with custom properties
legend = plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0.0), frameon=True, 
                    fontsize=32, prop={'weight': 'bold'})

# Make the legend visually larger and more prominent
legend.get_frame().set_linewidth(2)  # Thicker border
legend.get_frame().set_edgecolor('#333333')  # Darker border
plt.setp(legend.get_texts(), fontsize=30)  # Ensure text is large

# FIX: Use legend.get_lines() instead of legendHandles
for handle in legend.get_lines():
    handle.set_linewidth(6.0)  # Make lines thicker
    
# Adjust spacing between legend items for better visibility
plt.rcParams['legend.labelspacing'] = 1.2  # Vertical space between labels

# Adjust layout and save
plt.tight_layout()
plt.savefig('mlebench_vis.pdf', dpi=300, bbox_inches='tight', facecolor="white")