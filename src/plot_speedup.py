import matplotlib.pyplot as plt

# Replace these with actual times from your runs
sequential_time = 30.0  # Example: time from kmeans_seq.py
parallel_times = [20.0, 15.0, 10.0, 8.0]  # Example: times from kmeans_parallel.py
num_threads = [1, 2, 4, 8]

# Calculate speedup
speedup = [sequential_time / pt for pt in parallel_times]

# Plot speedup
plt.figure(figsize=(8, 5))
plt.plot(num_threads, speedup, marker='o')
plt.title('Speedup vs. Number of Threads')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.grid()
plt.show()
