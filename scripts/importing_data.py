import kagglehub
import os


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")

path = kagglehub.dataset_download(
    "uom190346a/sleep-health-and-lifestyle-dataset",
    output_dir=data_dir
)


