import kagglehub
import os
import shutil

datasets = [
    "uom190346a/sleep-health-and-lifestyle-dataset",
    "arashnic/fitbit"
]

data_dir = "data"

os.makedirs(data_dir, exist_ok=True)

for ds in datasets:
    print(f"Téléchargement du dataset : {ds}")
    src = kagglehub.dataset_download(ds)
    dataset_name = ds.split('/')[-1]
    dest = os.path.join(data_dir, dataset_name)
    if os.path.exists(dest):
        print(f"Suppression de l'ancien dossier : {dest}")
        shutil.rmtree(dest)
    shutil.move(src, dest)
    print(f"Dataset sauvegardé dans : {dest}\n")

