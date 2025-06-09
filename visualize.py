import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_vs_target(X, y, feature_name):
  plt.figure(figsize=(8,5))
  plt.scatter(X[feature_name], y, alpha=0.6, color='red')
  plt.xlabel(feature_name)
  plt.ylabel("Sale Price")
  plt.title(f"{feature_name} vs Sale Price")
  plt.grid(True)
  plt.tight_layout()
  plt.show()