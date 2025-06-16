import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import os


# Load CSV
data = np.loadtxt("./Charge_density/data/caffeine_coeffs.csv", delimiter=",")

# Ensure rows = points, columns = features
if data.shape[1] > data.shape[0]:
    data = data
else:
    data = data.T

# Create output directory
output_dir = "./Charge_density/plots/last_coeff/"
os.makedirs(output_dir, exist_ok=True)

# Generate 50 random thresholds between 0.2 and 2.0
#thresholds = np.random.uniform(1.0, 3, 30)

# Loop through thresholds
diagrams = ripser(data, thresh=0.4)['dgms']
plt.figure()
plot_diagrams(diagrams, show=False)
plt.title(f"Persistence Diagram")
plt.savefig(f"{output_dir}/diagram_thresh.png")
plt.close()

H1=diagrams[1]
print("H1features (birth,death):")
print(H1)








