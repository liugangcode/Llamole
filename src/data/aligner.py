# Copyright 2024 the LlamaFactory team.
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

import os
import re
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Union, Tuple

from datasets import Features

from ..extras.logging import get_logger
from .data_utils import Role

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .parser import DatasetAttr


logger = get_logger(__name__)

def extract_all_smiles(text):
    pattern = r'<mol_start>(.*?)<mol_end>'
    return re.findall(pattern, text)

def replace_all_smiles(text):
    pattern = r'<mol_start>.*?<mol_end>'
    return re.sub(pattern, '<molecule>', text)

def replace_smiles_with_callback(text):
    def replace_mol(match):
        design_end = match.group(1)
        smiles = match.group(2)
        # return f'{design_end}<molecule><callback_start>{smiles}<callback_end>'
        return f'{design_end}<molecule><rollback_start>{smiles}<rollback_end>'
    
    pattern = r'(<design_start><design_end>)<mol_start>(.*?)<mol_end>'
    text = re.sub(pattern, replace_mol, text)
    
    # Replace remaining molecules that are not immediately after <design_start><design_end>
    remaining_pattern = r'<mol_start>.*?<mol_end>'
    text = re.sub(remaining_pattern, '<molecule>', text)
    
    return text

def dict_to_list(data_dict, mol_properties):
    return [data_dict.get(prop, None) for prop in mol_properties]

def insert_bodies(text, num_insertions, retro_labels):
    design_pattern = r'<design_start>(.*?)<design_end>'
    retro_pattern = r'(This is step \d+ in the retrosynthesis process\..*?<retro_start>.*?<retro_end>)(.*?)(?=This is step \d+|$)'

    def replace_design(match):
        return f'<design_start>' + ''.join(['<design_body>'] * num_insertions) + f'<design_end>'
        
    def replace_retro(match, label):
        step_content = match.group(1)
        remaining_text = match.group(2)
        retro_match = re.search(r'<retro_start>(.*?)<retro_end>', step_content)
        if retro_match and label is not None:
            modified_content = f'<retro_start>' + ''.join(['<retro_body>'] * num_insertions) + f'<retro_end>'
            return re.sub(r'<retro_start>.*?<retro_end>', modified_content, step_content)
        return step_content + remaining_text
                
    text = re.sub(design_pattern, replace_design, text)
    
    steps = re.finditer(retro_pattern, text)
    modified_text = ""
    last_end = 0
    
    for i, step in enumerate(steps):
        label = retro_labels[i] if i < len(retro_labels) else None
        modified_text += text[last_end:step.start()] + replace_retro(step, label)
        last_end = step.end()
    
    modified_text += text[last_end:]
    return modified_text

def extract_retro_products(text):
    pattern = r'<retro_end>(.*?)>>'
    matches = re.findall(pattern, text)
    return [match.strip() for match in matches]
    
def convert_molqa(
    examples: Dict[str, List[Any]], dataset_attr: "DatasetAttr", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    r"""
    Converts alpaca format dataset to the standard format.
    """
    outputs = {"prompt": [], "response": [], "system": [], "molecules": [], "property": [], "retro_labels": [], "retro_products": []}

    mol_properties = ['BBBP', 'HIV', 'BACE', 'CO2', 'N2', 'O2', 'FFV', 'TC', 'SC', 'SA']
    for i in range(len(examples[dataset_attr.prompt])):
        prompt = []
        if dataset_attr.history and isinstance(examples[dataset_attr.history][i], list):
            for old_prompt, old_response in examples[dataset_attr.history][i]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        content = []
        if dataset_attr.prompt and examples[dataset_attr.prompt][i]:
            content.append(examples[dataset_attr.prompt][i])

        if dataset_attr.query and examples[dataset_attr.query][i]:
            content.append(examples[dataset_attr.query][i])

        prompt.append({"role": Role.USER.value, "content": "\n".join(content)})  # "prompt\nquery"

        if dataset_attr.response and isinstance(examples[dataset_attr.response][i], str):  # normal example
            current_response = examples[dataset_attr.response][i]
            smiles_list = extract_all_smiles(current_response)
            modified_response = replace_smiles_with_callback(current_response)
            retro_labels = examples[dataset_attr.retro][i] if dataset_attr.retro else []
            retro_products = extract_retro_products(current_response)
            modified_response = insert_bodies(modified_response, data_args.learned_query_size, retro_labels)
            # modified_response = insert_bodies(modified_response, dataset_attr.learned_query_size, retro_labels)
            response = [{"role": Role.ASSISTANT.value, "content": modified_response}]
        else:  # unsupervised
            response = []

        outputs["prompt"].append(prompt)
        outputs["response"].append(response)
        outputs["system"].append(examples[dataset_attr.system][i] if dataset_attr.system else "")
        outputs["molecules"].append(smiles_list)
        outputs["property"].append(dict_to_list(examples[dataset_attr.property][i], mol_properties))
        outputs["retro_labels"].append(retro_labels)
        outputs["retro_products"].append(retro_products)
        
    return outputs

def map_smiles_to_id(example, smiles_to_id):
    example['molecules'] = [smiles_to_id[smiles] for smiles in example['molecules']]
    return example

def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Tuple[Union["Dataset", "IterableDataset"], Dict[int, str]]:
    r"""
    Aligns the dataset and maps unique SMILES strings to molecule IDs.

    This function performs the following operations:
    1. Converts the dataset to the required format (molqa).
    2. Extracts all unique SMILES strings from the dataset.
    3. Maps each unique SMILES string to a unique integer ID (0, 1, 2, ...).
    4. Update 'molecules' field to each example, containing the mapped IDs.

    The aligned dataset contains the following fields:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        system: "..."
        molecules: [List of SMILES string]
        property: [List of float values]
        retro_labels: [List of int values]
        retro_products: [List of SMILES string]

    Args:
        dataset (Union["Dataset", "IterableDataset"]): The input dataset.
        dataset_attr (DatasetAttr): Attributes of the dataset.
        data_args (DataArguments): Arguments for data processing.
        training_args (Seq2SeqTrainingArguments): Arguments for training.

    Returns:
        Tuple[Union["Dataset", "IterableDataset"], Dict[int, str]]: 
            - The aligned and converted dataset with molecule IDs.
            - A dictionary mapping molecule IDs to their SMILES strings.
    """
    assert dataset_attr.formatting == "molqa"

    features = Features.from_dict(
        {
            "prompt": [
                {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
            ],
            "response": [
                {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
            ],
            "system": {"dtype": "string", "_type": "Value"},
            "molecules": [{'dtype': "string", "_type": "Value"}],
            "property": [{"dtype": "float", "_type": "Value"}],
            "retro_labels": [{"dtype": "int32", "_type": "Value"}],
            "retro_products": [{'dtype': "string", "_type": "Value"}],
        }
    )
    
    convert_func = partial(convert_molqa, dataset_attr=dataset_attr, data_args=data_args)
    aligned = dataset.map(
        convert_func,
        batched=True,
        remove_columns=['instruction', 'input', 'output', 'property', 'retro'],
        features=features,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
        desc="Converting molqa format of dataset"
    )
    
    # Extract all unique SMILES strings and map them to molecule IDs
    all_smiles = set()
    for item in aligned:
        all_smiles.update(item['molecules'])
        all_smiles.update(item['retro_products'])
    
    smiles_to_id = {smiles: idx for idx, smiles in enumerate(sorted(all_smiles))}
    id_to_smiles = {idx: smiles for smiles, idx in smiles_to_id.items()}
    
    def map_smiles_to_id(example, smiles_to_id):
        example['molecules'] = [smiles_to_id[smiles] for smiles in example['molecules']]
        example['retro_products'] = [smiles_to_id[smiles] for smiles in example['retro_products']]
        return example
    
    smiles_convert_func = partial(map_smiles_to_id, smiles_to_id=smiles_to_id)

    aligned = aligned.map(
        smiles_convert_func,
        desc="Mapping SMILES to molecule IDs",
    )
    
    return aligned, id_to_smiles