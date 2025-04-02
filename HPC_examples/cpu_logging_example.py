import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="logs/cpu_logging_example.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def expensive_task(task_id):
    logging.info(f"Starting task {task_id}")
    total = 0
    for i in range(10**6):
        total += i
        if i % 200000 == 0:
            logging.debug(f"Task {task_id}: progress {i/10**6:.0%}")
    logging.info(f"Completed task {task_id} with result {total}")
    return total

def main():
    start_time = datetime.now()
    logging.info("Job started.")

    for i in range(2):  # Simulate two tasks
        result = expensive_task(i)
        logging.debug(f"Result from task {i}: {result}")

    logging.info("Job finished.")
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"Total runtime: {duration}")

if __name__ == "__main__":
    main()
