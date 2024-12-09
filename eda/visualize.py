import cv2 # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as patches # type: ignore


def visualize_bbox(ax, x, y, w, h, label, color):
    """Visualizes a bounding box on a matplotlib axes object.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to draw the bounding box on.
        x (float): The x-coordinate of the top-left corner of the bounding box.
        y (float): The y-coordinate of the top-left corner of the bounding box.
        w (float): The width of the bounding box.
        h (float): The height of the bounding box.
        label (str): The label to display on the bounding box.
        color (str or tuple): The color of the bounding box and label.

    Returns:
        matplotlib.axes.Axes: The matplotlib axes object with the bounding box added.

    Example:
        >>> fig, ax = plt.subplots(1)
        >>> ax = visualize_bbox(ax, 10, 10, 100, 100, 'person', 'red')
        >>> plt.show()
    """
    rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, label, color='white', fontsize=5, backgroundcolor=color)
    return ax 



