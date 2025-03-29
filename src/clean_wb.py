import wandb

api = wandb.Api()
project = "7shoe/domShift-extensive"

runs = list(api.runs(project))
runs = runs[::-1]

for run in runs:
    for f in run.files():
        # delete every .pth except epoch10
        if f.name.endswith(".pth") and "epoch10" not in f.name:
            print(f"Deleting {run.id}/{f.name}")
            f.delete()