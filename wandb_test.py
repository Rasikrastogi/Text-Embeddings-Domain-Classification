import wandb
import time

# Start a new run
wandb.init(project="domain-classification", name="wandb-demo-run")

# Log a few metrics step by step
for step in range(5):
    acc = 0.8 + step * 0.02   # fake accuracy improving
    loss = 0.5 - step * 0.05  # fake loss decreasing

    wandb.log({"step": step, "accuracy": acc, "loss": loss})
    print(f"Step {step}: accuracy={acc:.2f}, loss={loss:.2f}")
    time.sleep(1)

print("\n✅ Test finished! Check your W&B dashboard for 'wandb-demo-run'.")
