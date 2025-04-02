import logging
import os
from datetime import datetime
from multiprocessing import Pool

# Setup logging
logging.basicConfig(
    filename="logs/cpu_multiproc_example.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
)

def get_cpu_info():
    pid = os.getpid()
    try:
        with open("/proc/self/stat", "r") as f:
            stat_fields = f.readline().split()
            cpu_id = stat_fields[38]  # field 39 is CPU number
    except Exception as e:
        cpu_id = "unknown"
        logging.warning(f"Could not get CPU info: {e}")
    return f"PID {pid}, CPU {cpu_id}"

def expensive_task(task_id):
    cpu_info = get_cpu_info()
    logging.info(f"Task {task_id} started on {cpu_info}.")
    total = 0
    for i in range(10**6):
        total += i
        if i % 200000 == 0:
            logging.debug(f"Task {task_id} (on {cpu_info}): progress {i / 10**6:.0%}")
    logging.info(f"Task {task_id} finished on {cpu_info} with result {total}")
    return total

def main():
    start_time = datetime.now()
    logging.info("Main job started.")
    
    num_tasks = 4  # Match --cpus-per-task
    logging.info(f"Spawning {num_tasks} tasks with multiprocessing.")

    with Pool(processes=num_tasks) as pool:
        results = pool.map(expensive_task, range(num_tasks))

    for i, result in enumerate(results):
        logging.debug(f"Result from task {i}: {result}")

    logging.info("Main job finished.")
    logging.info(f"Total runtime: {datetime.now() - start_time}")

if __name__ == "__main__":
    main()
