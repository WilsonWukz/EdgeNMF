import time
import tracemalloc

def measure_performance(algo_func, *args, **kwargs):
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.start()

    start_time = time.time()
    result = algo_func(*args, **kwargs)
    end_time = time.time()

    # Get the peak value
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    time_cost = end_time - start_time
    peak_memory_mb = peak / (1024 * 1024)
    return result, time_cost, peak_memory_mb
