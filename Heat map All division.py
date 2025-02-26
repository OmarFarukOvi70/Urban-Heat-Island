import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Corrected correlation data for all cities (lower triangle format)
correlation_data = {
    'A': [
        [1.0, 0.907, -0.987, -0.140, 0.717],
        [0.907, 1.0, -0.841, -0.050, 0.812],
        [-0.987, -0.841, 1.0, 0.173, -0.645],
        [-0.140, -0.050, 0.173, 1.0, 0.164],
        [0.717, 0.812, -0.645, 0.164, 1.0]
    ],
    'B': [
        [1.0, 0.981, -0.986, 0.090, 0.903],
        [0.981, 1.0, -0.950, 0.107, 0.848],
        [-0.986, -0.950, 1.0, -0.025, -0.917],
        [0.090, 0.107, -0.025, 1.0, 0.037],
        [0.903, 0.848, -0.917, 0.037, 1.0]
    ],
    'C': [
        [1.0, 0.926, -0.993, -0.013, 0.859],
        [0.926, 1.0, -0.898, -0.069, 0.812],
        [-0.993, -0.898, 1.0, -0.037, -0.862],
        [-0.013, -0.069, -0.037, 1.0, 0.061],
        [0.859, 0.812, -0.862, 0.061, 1.0]
    ],
    'D': [
        [1.0, 0.962, -0.930, 0.780, 0.106],
        [0.962, 1.0, -0.830, 0.849, -0.016],
        [-0.930, -0.830, 1.0, -0.654, -0.380],
        [0.780, 0.849, -0.654, 1.0, 0.040],
        [0.106, -0.016, -0.380, 0.040, 1.0]
    ],
    'E': [
        [1.0, 0.989, -0.962, 0.577, 0.329],
        [0.989, 1.0, -0.933, 0.551, 0.314],
        [-0.962, -0.933, 1.0, -0.608, -0.293],
        [0.577, 0.551, -0.608, 1.0, 0.469],
        [0.329, 0.314, -0.293, 0.469, 1.0]
    ],
    'F': [
        [1.0, 0.976, -0.963, 0.385, 0.566],
        [0.976, 1.0, -0.905, 0.285, 0.496],
        [-0.963, -0.905, 1.0, -0.458, -0.691],
        [0.385, 0.285, -0.458, 1.0, 0.280],
        [0.566, 0.496, -0.691, 0.280, 1.0]
    ],
    'G': [
        [1.0, 0.937, -0.976, 0.388, 0.543],
        [0.937, 1.0, -0.855, 0.239, 0.420],
        [-0.976, -0.855, 1.0, -0.444, -0.588],
        [0.388, 0.239, -0.444, 1.0, 0.239],
        [0.543, 0.420, -0.588, 0.239, 1.0]
    ],
    'H': [
        [1.0, 0.988, -0.925, 0.017, 0.213],
        [0.988, 1.0, -0.869, 0.006, 0.265],
        [-0.925, -0.869, 1.0, -0.118, -0.045],
        [0.017, 0.006, -0.118, 1.0, -0.719],
        [0.213, 0.265, -0.045, -0.719, 1.0]
    ]
}

parameters = ['Ambient temp', 'Feel temp', 'Humidity', 'Wind speed', 'Air quality']

for city, data in correlation_data.items():
    df = pd.DataFrame(data, index=parameters, columns=parameters)

    # Mask upper triangle (keep lower triangle + diagonal)
    mask = np.triu(np.ones_like(df, dtype=bool), k=1)

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        df,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        annot_kws={'size': 10, 'weight': 'bold'}  # Adjust annotation size
    )

    # Boldface annotations for significant values (example: |corr| > 0.5)
    for text in heatmap.texts:
        val = float(text.get_text())
        if abs(val) >= 0.5:
            text.set_weight('bold')
            text.set_size(12)

    plt.title(f'{city} Correlation Matrix', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
