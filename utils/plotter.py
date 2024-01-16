import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from config.settings import GlulamConfig
import seaborn as sns
import pandas as pd


def get_press_layout(press, filename=None):
    """
    Generates the layout as rectangles for the given press.

    Parameters:
    - press (GlulamPackagingProcessor): The press object to be plotted.
    - filename (str): The filename to save the rectangles to. If None, the data frame is not saved.

    Returns:
    - rects (pandas.DataFrame): A dataframe containing the coordinates of the rectangles in the press.
    """
    # Create a pandas dataframe for the rectangles
    rects = pd.DataFrame(columns=['k', 'r', 'x', 'y', 'w', 'h', 'type', 'sub_type', 'order'])

    # Plot each group
    for k in press.K:
        y_pos = 0  # Initialize y position for vertical stacking
        for r in press.R:
            # Draw the actual Lp in gray
            rects = rects._append({'k': k, 'r': r, 'x': GlulamConfig.MAX_ROLL_WIDTH, 'y': y_pos,
                                   'w': -press.Lp_estimated[k][r], 'h': press.h[k][r], 'type': 'Lp',
                                   'sub_type': 'estimated'}, ignore_index=True)

            # Draw the estimated Lp in white
            rects = rects._append({'k': k, 'r': r, 'x': GlulamConfig.MAX_ROLL_WIDTH, 'y': y_pos,
                                   'w': -press.Lp_actual[k][r], 'h': press.h[k][r], 'type': 'Lp',
                                   'sub_type': 'actual'}, ignore_index=True)

            # All patterns used in the press
            patterns = [{
                'id': j,
                'pattern': [(i, press.A[i][j]) for i in press.I if press.A[i][j] > 0],
                'height': press.H[j] / GlulamConfig.LAYER_HEIGHT,
                'width': press.L[j],
                'repeat': int(press.xn[j][k][r])}
                for j in press.J if press.x[j][k][r]]

            # order by descending width
            patterns = sorted(patterns, key=lambda x: x['width'], reverse=True)

            # Start from the top of the previous group (or 0 if first group)
            y_pos = 0 if r == 0 else press.h[k][r - 1]
            for pattern in patterns:
                # Draw rectangles for each cutting pattern, stacking up vertically and right aligned to the roll width
                for repeat in range(pattern['repeat']):
                    rects = rects._append({'k': k, 'r': r, 'x': GlulamConfig.MAX_ROLL_WIDTH, 'y': y_pos,
                                           'w': -pattern['width'], 'h': pattern['height'], 'type': 'pattern',
                                           'sub_type': pattern['id']}, ignore_index=True)

                    x_pos = GlulamConfig.MAX_ROLL_WIDTH  # Initialize x position for horizontal stacking
                    for (item, item_count) in pattern['pattern']:
                        # Draw the actual item
                        for _ in range(item_count):
                            rects = rects._append({'k': k, 'r': r, 'x': x_pos, 'y': y_pos,
                                                   'w': -press.patterns.data.widths[item],
                                                   'h': press.patterns.data.layers[item], 'type': 'item',
                                                   'sub_type': item, 'order': press.patterns.data.order[item]},
                                                  ignore_index=True)

                            assert press.patterns.data.layers[item] == pattern['height'], \
                                f"Layer height mismatch: {press.patterns.data.layers[item]} != {pattern['height']}"
                            x_pos -= press.patterns.data.widths[item]  # Update x position for next rectangle

                    y_pos += pattern['height']  # Update y position for next rectangle

    if filename is not None:
        rects.to_csv(filename, index=False)

    return rects


def plot_rectangles(rects, filename=None):
    """
    Plots rectangles from a DataFrame and saves the plot to a file if a filename is provided.

    Args:
    - rects (pandas.DataFrame): The DataFrame containing the rectangles' coordinates and properties.
    - filename (str, optional): The filename to save the plot to. If None, the plot is not saved.
    """

    # Calculate layout based on unique 'k' values in the DataFrame
    unique_presses = rects['k'].unique()
    num_presses = len(unique_presses)
    num_cols = math.ceil(math.sqrt(num_presses))
    num_rows = math.ceil(num_presses / num_cols)
    color = sns.color_palette("hls", len(rects['sub_type'].unique())).as_hex()

    # Create figure and axes
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 3 * num_rows))
    if num_presses == 1:
        axs = [axs]  # Ensure axs is iterable even for a single subplot
    axs = axs.flatten()

    # Plot rectangles for each press
    for ax, k in zip(axs, unique_presses):
        press_rects = rects[rects['k'] == k]
        for _, row in press_rects.iterrows():
            rect = patches.Rectangle(
                (row['x'], row['y']), row['w'], row['h'],
                facecolor='gray' if row['type'] == 'Lp' else (
                    color[row['sub_type']] if row['type'] == 'item' else 'white'),
                edgecolor='black', linewidth=1)
            ax.add_patch(rect)

        # Configure axes
        ax.set_title(f"Press: {k + 1}")
        ax.set_xlim(0, GlulamConfig.MAX_ROLL_WIDTH + 1)
        ax.set_ylim(0, GlulamConfig.MAX_HEIGHT_LAYERS + 1)
        ax.set_xticks([])  # Disable x-axis ticks
        ax.set_yticks([])  # Disable y-axis ticks

    plt.tight_layout()  # Fit the plot to the figure

    if filename:
        plt.savefig(filename)  # Save the figure

    plt.close(fig)  # Close the figure to free memory
