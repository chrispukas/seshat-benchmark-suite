import numpy as np

from typing import List, Dict, Any, Optional, Tuple
from matplotlib import pyplot as plt

def plot_stacked_bar_chart(data: Dict[str, Tuple[str, int]],
                           colours: Optional[List[str]] = ['#e377c2', '#7bac4e', '#d4c04c', '#4c70c4', '#8c564b', '#e7c377', '#c7c7c7'], 
                           title: Optional[str] = "", 
                           legend_title: Optional[str] = "",
                           x_label: Optional[str] = "",
                           y_label: Optional[str] = "", 
                           figsize: Optional[Tuple[int, int]] = (4,6),
                           y_limits: Optional[Tuple[float, float]] = (0, 100),
                           absolute: bool = False
                           ) -> None:
    plt.figure(figsize=figsize)


    categories = list(data.keys())
    for i, category in sorted(enumerate(categories)):
        sub_categories, counts = zip(*data[category])
        sum_counts: int = np.sum(counts)
        print(category, sum_counts)
        bottom: float = 0.0

        for j, (sub_category, count) in \
            enumerate(zip(sub_categories, counts)):
            
            if absolute:
                height: float = count
            else:    
                fraction: float = (count / sum_counts) if sum_counts > 0 else 0
                height: float = fraction * 100

            plt.bar(
                category,
                height=height,
                bottom=bottom,
                color=colours[j % len(colours)],
                label=sub_category if i == 0 else "",
            )

            bottom += height

    plt.yticks(np.arange(y_limits[0], y_limits[1] + 1, 10))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(y_limits)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title=legend_title)
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.show()