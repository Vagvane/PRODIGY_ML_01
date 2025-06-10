import matplotlib.pyplot as plt


def plot_all_features(X, y):
    readable_labels = {
        'GrLivArea': 'Living Area (sqft)',
        'BedroomAbvGr': 'Number of Bedrooms',
        'FullBath': 'Number of Bathrooms'
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    features = list(readable_labels.keys())

    for i, feature in enumerate(features):
        label = readable_labels[feature]
        axs[i].scatter(X[feature], y)
        axs[i].set_xlabel(label)
        axs[i].set_ylabel("House Price ($)")
        axs[i].set_title(f"{label} vs House Price")

    plt.tight_layout()
    plt.savefig("output_graphs.png")
    plt.show()
