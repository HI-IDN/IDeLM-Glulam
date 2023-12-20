import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from config.settings import GlulamConfig


def plot_cutting_patterns(df, filename):
    """
    Generates a plot of cutting patterns grouped by width and saves it to a file.

    Parameters:
    - df (DataFrame): DataFrame containing cutting pattern data with columns 'totlen', 'height', 'width', 'lengths'.
    - filename (str): The path and name of the file where the plot will be saved.
    """
    # Sort and group the DataFrame
    df_sorted = df.sort_values(by=['totlen', 'height'])
    grouped = df_sorted.groupby('width')

    # Create figure and axes
    fig, axs = plt.subplots(len(grouped), 1, figsize=(10, 6 * len(grouped)))

    # Plot each group
    for ax, (width, group) in zip(axs, grouped):
        y_pos = 0  # Initialize y position for vertical stacking
        for _, row in group.iterrows():
            # Draw rectangles for each cutting pattern
            rect = patches.Rectangle((0, y_pos), row['totlen'], row['height'], facecolor='white', edgecolor='black',
                                     linewidth=1)
            ax.add_patch(rect)

            # Add vertical lines for cutting patterns
            cuts = np.cumsum(np.sort(row['lengths']))  # Cumulative sum of sorted lengths
            for cut in cuts:
                ax.vlines(x=cut, ymin=y_pos, ymax=y_pos + row['height'], color='blue', linestyle='-')

            y_pos += row['height']  # Update y position

        # Configure axes
        ax.set_title(f"Width: {width}")
        ax.set_xlim(0, GlulamConfig.MAX_ROLL_WIDTH)
        ax.set_ylim(0, y_pos + 10)  # Adding padding to y-axis
        ax.set_yticks(range(0, int(y_pos + 10), GlulamConfig.MAX_HEIGHT_LAYERS + 1))
        ax.grid(True, which='both', axis='y')

        # Optional: Add additional lines (e.g., ax.axvline) here based on your requirements

    plt.tight_layout()
    plt.savefig(filename)  # Save the figure
    plt.close(fig)  # Close the figure to free memory
