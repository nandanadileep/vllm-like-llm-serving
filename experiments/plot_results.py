import matplotlib.pyplot as plt

concurrency = [1, 2, 4, 8, 16, 32]

avg_latency = [0.0777, 0.0997, 0.0932, 0.1614, 0.3008, 0.5249]
max_latency = [0.0833, 0.1043, 0.0976, 0.1735, 0.3387, 0.6766]
min_latency = [0.0725, 0.0939, 0.0878, 0.0888, 0.0864, 0.0929]
avg_wait_time = 0.13621963921377755
max_queue_length = 28

# Plot 1: Avg Latency vs Concurrency
plt.figure()
plt.plot(concurrency, avg_latency, marker="o")
plt.xlabel("Concurrency")
plt.ylabel("Average Latency (s)")
plt.title("Average Latency vs Concurrency")
plt.grid()
plt.savefig("avg_latency.png")
plt.clf()

# Plot 2: Latency spread
plt.figure()
plt.plot(concurrency, min_latency, marker="o", label="Min Latency")
plt.plot(concurrency, avg_latency, marker="o", label="Avg Latency")
plt.plot(concurrency, max_latency, marker="o", label="Max Latency")
plt.xlabel("Concurrency")
plt.ylabel("Latency (s)")
plt.title("Latency Distribution vs Concurrency")
plt.legend()
plt.grid()
plt.savefig("latency_distribution.png")
plt.clf()

# Plot 3: Growth curve (log-like behavior)
plt.figure()
plt.plot(concurrency, avg_latency, marker="o")
plt.xlabel("Concurrency")
plt.ylabel("Latency (s)")
plt.title("Latency Growth Under Load")
plt.xscale("log")
plt.grid()
plt.savefig("latency_log_scale.png")
plt.clf()

# Plot 4: Queue insight summary
plt.figure()
plt.bar(["avg_wait_time", "max_queue_length"], [avg_wait_time, max_queue_length])
plt.ylabel("Value")
plt.title("Queue Insight Summary")
plt.grid(axis="y")
plt.savefig("queue_insights.png")
plt.clf()
