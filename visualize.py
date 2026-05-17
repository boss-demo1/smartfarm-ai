import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("smartfarm_dataset.csv")

# Temperature graph
plt.figure(figsize=(10,5))
plt.plot(df["temperature"])
plt.title("Temperature")
plt.xlabel("Reading")
plt.ylabel("°C")
plt.grid(True)
plt.show()

# Humidity graph
plt.figure(figsize=(10,5))
plt.plot(df["humidity"])
plt.title("Humidity")
plt.xlabel("Reading")
plt.ylabel("%")
plt.grid(True)
plt.show()

# LDR graph
plt.figure(figsize=(10,5))
plt.plot(df["ldr_value"])
plt.title("LDR Value")
plt.xlabel("Reading")
plt.ylabel("Light")
plt.grid(True)
plt.show()
