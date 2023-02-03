from matplotlib import pyplot as plt

from experiment2 import mse_test, mse_train, distance_test, distance_train


fig = plt.figure()
ax = fig.add_subplot()
ax.plot([mt ** 2 for mt in mse_test], color="b", label="Test")
ax.plot([mt ** 2 for mt in mse_train], color="r", label="Train")
ax.set(xlabel="Epoch", ylabel="MSE")

plt.legend()
plt.title("MSE across training epochs")

plt.show()

mean = 55.75
random_forest = 48.11
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(
    [30, len(distance_train)],
    [mean] * 2,
    color="black",
    label="Mean",
)
ax.plot(
    [30, len(distance_train)],
    [random_forest] * 2,
    color="Grey",
    label="Random Forest",
)
ax.plot(distance_test, color="b", label="Test")
ax.plot(distance_train, color="r", label="Train")
ax.text(len(distance_train) - 6, mean - 2.5, f"{mean}")
ax.text(len(distance_train) - 6, random_forest - 2.5, f"{random_forest}")
ax.text(
    len(distance_test) - 6, distance_test[-1] - 3.5, f"{round(distance_test[-1], 2)}"
)
ax.text(
    len(distance_train) - 6, distance_train[-1] - 2.5, f"{round(distance_train[-1], 2)}"
)
ax.set(xlabel="Epoch", ylabel="Distance")

plt.legend(loc="upper left")
plt.title("Distance from correct across training epochs")

plt.show()
