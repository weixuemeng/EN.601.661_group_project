import os
from huggingface_hub import snapshot_download

# download the linemod dataset
dataset_name = "lm"
local_dir = "./datasets"

# create the local directory if it does not exist
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

snapshot_download(repo_id="bop-benchmark/datasets", 
                  allow_patterns=f"{dataset_name}/*zip",
                  repo_type="dataset", 
                  local_dir=local_dir)

