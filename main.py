# Copyright 2024 Llamole Team
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
from src.train.tuner import run_train, merge_adapter
from src.eval.workflow import run_eval
from tqdm import tqdm

from huggingface_hub import hf_hub_download

def download_data():
    repo_id = "liuganghuggingface/Llamole-MolQA"
    files_to_download = [
        "molqa_drug.json",
        "molqa_material.json",
        "molqa_train.json"
    ]
    local_dir = "data"
    
    # Create the data directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading files from {repo_id} to {local_dir}/")
    for file in tqdm(files_to_download, desc="Downloading files"):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded: {file}")
        except Exception as e:
            print(f"Error downloading {file}: {e}")
    
    print("Download complete!")

if __name__ == "__main__":
    command = sys.argv.pop(1) if len(sys.argv) != 1 else 'train'
    if command == 'train':
        run_train()
    elif command == 'export':
        merge_adapter()
    elif command == 'eval':
        run_eval()
    elif command == 'download_data':
        download_data()
    else:
        print(f"Invalid command: {command}. Please use 'train', 'export', 'eval', or 'download_data'.")
        sys.exit(1)