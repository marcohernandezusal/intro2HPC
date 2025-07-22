import csv

best_acc = -1.0
best_entry = None

with open("mnist_outputs/grid_results.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        job_id, lr, batch_size, acc, model_path = row
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_entry = {
                "job_id": job_id,
                "lr": lr,
                "batch_size": batch_size,
                "accuracy": acc,
                "model_path": model_path,
            }

print("🏆 Best Model Found:")
print(best_entry)