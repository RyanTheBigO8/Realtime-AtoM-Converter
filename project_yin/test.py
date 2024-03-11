import matplotlib.pyplot as plt
import numpy as np

# Generate some random data for demonstration
data = np.random.randn(1000)

# Create a histogram with spaces between bars
plt.hist(data, bins=30, rwidth=0.8, edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram with Spaces Between Bars')

# Show the plot
plt.show()
