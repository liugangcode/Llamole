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

import torch
from torch.utils.data import Dataset

from ..extras.constants import BOND_INDEX

def dict_to_list(data_dict, mol_properties):
    return [data_dict.get(prop, float("nan")) for prop in mol_properties]

class MolQADataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mol_properties = [
            "BBBP",
            "HIV",
            "BACE",
            "CO2",
            "N2",
            "O2",
            "FFV",
            "TC",
            "SC",
            "SA",
        ]
        item = self.data[idx]
        instruction = item["instruction"]
        input_text = item["input"]
        property_data = dict_to_list(item["property"], mol_properties)
        property_data = torch.tensor(property_data)

        # Combine instruction and input
        combined_input = f"{instruction}\n{input_text}"

        # Create messages for chat template
        messages = [
            {"role": "user", "content": combined_input}
        ]

        # Apply chat template
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize the chat text
        encoding = self.tokenizer(
            chat_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        
        return {
            "input_ids": encoding.input_ids.squeeze(),
            "attention_mask": encoding.attention_mask.squeeze(),
            "property": property_data,
        }