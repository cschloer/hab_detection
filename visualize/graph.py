import numpy as np
from matplotlib import pyplot as plt

from dist import dist

fig, ax = plt.subplots()
color = ["green"] * 100 + ["orange"] * 100 + ["red"] * 54
low = [149, 149, 149, 255]
# moderate =
ax.plot(range(100), dist[:100], color=low)
ax.plot(range(100, 200), dist[100:200], color="orange")
ax.plot(range(200, 254), dist[200:254], color="red")
ax.set_yscale("log")
ax.set_ylabel("Distribution")
ax.set_xlabel("HAB Value")
plt.show()
