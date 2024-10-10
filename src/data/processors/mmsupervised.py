# Copyright 2024 Llamole Team
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.constants import IGNORE_INDEX, BOND_INDEX, NO_LABEL_INDEX
from ...extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template

import os
from rdkit import Chem
import torch
from torch_geometric.data import Data, Batch
import pickle

logger = get_logger(__name__)

import os
import torch
from typing import Dict
from torch_geometric.data import Data
from rdkit import Chem
import pickle


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    new_source_len = max(cutoff_len - new_target_len, 0)
    return new_source_len, new_target_len

def encode_graph_pyg(
    data_path: Optional[str] = None, mol_id_to_smiles: Optional[Dict[str, str]] = None
) -> Dict[str, Data]:
    """
    Converts molecule data to a dictionary of PyTorch Geometric Data objects, with caching functionality.
    Uses a sparse representation for efficiency.

    Args:
        data_path (Optional[str]): Path to the Hugging Face dataset folder.
        mol_id_to_smiles (Optional[Dict[str, str]]): Dictionary where keys are molecule IDs
                                                     and values are SMILES strings.

    Returns:
        Dict[str, Data]: Dictionary where keys are molecule IDs and values are
                         PyTorch Geometric Data objects.

    Raises:
        ValueError: If both data_path and mol_id_to_smiles are None, or if data_path is provided but loading fails.
    """
    print(f"Current execution directory: {os.getcwd()}")

    if data_path is None and mol_id_to_smiles is None:
        raise ValueError("Either data_path or mol_id_to_smiles must be provided.")

    if data_path is not None:
        cache_file = os.path.join(data_path, "pyg_molecule.pickle")

        # Try to load cached data
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load cached data: {e}")

    mol_id_to_pyg = {}

    for mol_id, smiles in mol_id_to_smiles.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string for molecule {mol_id}: {smiles}")

        type_idx = []
        heavy_atom_indices = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:  # Exclude hydrogen atoms
                type_idx.append(
                    119 - 2 if atom.GetSymbol() == "*" else atom.GetAtomicNum() - 2
                )
                heavy_atom_indices.append(atom.GetIdx())

        x = torch.LongTensor(type_idx)

        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if start in heavy_atom_indices and end in heavy_atom_indices:
                start_new, end_new = heavy_atom_indices.index(
                    start
                ), heavy_atom_indices.index(end)
                edge_index.extend([[start_new, end_new], [end_new, start_new]])
                bond_type = BOND_INDEX[bond.GetBondType()]
                edge_attr.extend([bond_type, bond_type])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        mol_id_to_pyg[mol_id] = data

    # Save cached data if data_path is provided
    if data_path is not None:
        with open(cache_file, "wb") as f:
            pickle.dump(mol_id_to_pyg, f)

        print(f"Saved PyG data to {cache_file}")

    return mol_id_to_pyg

def encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    molecule_ids: List[int],
    retro_product_ids: List[int],
    retro_labels: List[int],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:

    messages = prompt + response
    input_ids, labels = [], []
    final_molecule_ids = []
    final_product_ids = []
    final_retro_labels = []

    encoded_pairs = template.encode_multiturn(tokenizer, messages, system)
    special_tokens = [
        "<design_start>",
        "<design_end>",
        "<design_body>",
        "<molecule>",
        "<retro_start>",
        "<retro_end>",
        "<retro_body>",
    ]
    special_token_ids = template._convert_elements_to_ids(tokenizer, special_tokens)
    special_token_dict = dict(zip(special_tokens, special_token_ids))

    total_length = 1 if template.efficient_eos else 0
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= data_args.cutoff_len:
            break

        source_len, target_len = infer_seqlen(
            len(source_ids), len(target_ids), data_args.cutoff_len - total_length
        )
        source_ids = source_ids[:source_len]

        # Ensure balanced retro tags when truncating
        retro_start_indices = [
            i
            for i, id in enumerate(target_ids)
            if id == special_token_dict["<retro_start>"]
        ]
        retro_end_indices = [
            i
            for i, id in enumerate(target_ids)
            if id == special_token_dict["<retro_end>"]
        ]

        if retro_start_indices and retro_end_indices:
            # Find the last matching pair that fits within target_len
            last_pair_index = -1
            for start, end in zip(retro_start_indices, retro_end_indices):
                if end < target_len:
                    last_pair_index = end
                else:
                    break

            if last_pair_index >= 0:
                target_len = last_pair_index + 1
            else:
                # If no complete pair fits, truncate before the first start tag
                target_len = (
                    min(target_len, retro_start_indices[0])
                    if retro_start_indices
                    else target_len
                )

        target_ids = target_ids[:target_len]

        # Calculate the number of molecules in this turn
        molecules_in_turn = target_ids.count(special_token_dict["<molecule>"])
        retro_start_in_turn = target_ids.count(special_token_dict["<retro_start>"])
        retro_end_in_turn = target_ids.count(special_token_dict["<retro_end>"])

        assert retro_start_in_turn == retro_end_in_turn

        retro_product_ids_in_turn = retro_product_ids[:retro_end_in_turn]
        retro_labels_in_turn = retro_labels[:retro_end_in_turn]

        # Add corresponding retro_labels and retro_product_ids
        final_molecule_ids.extend(molecule_ids[:molecules_in_turn])
        final_product_ids.extend(retro_product_ids_in_turn)
        final_retro_labels.extend(retro_labels_in_turn)

        total_length += source_len + target_len

        if data_args.train_on_prompt:
            source_mask = source_ids
        elif turn_idx != 0 and template.efficient_eos:
            source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (
                len(source_ids) - 1
            )
        else:
            source_mask = [IGNORE_INDEX] * len(source_ids)

        source_mask = [
            IGNORE_INDEX if id in special_token_dict.values() else id
            for id in source_mask
        ]
        target_ids_mask = [
            id if id in [special_token_dict["<retro_start>"], special_token_dict["<design_start>"]]
            else (IGNORE_INDEX if id in special_token_dict.values() else id)
            for id in target_ids
        ]

        input_ids += source_ids + target_ids
        labels += source_mask + target_ids_mask

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return input_ids, labels, final_molecule_ids, final_product_ids, final_retro_labels


def preprocess_mmsupervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "molecule_ids": [],
        "molecule_properties": [],
        "retro_labels": [],
        "retro_product_ids": [],
    }

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning(
                "Dropped invalid example: {}".format(
                    examples["prompt"][i] + examples["response"][i]
                )
            )
            continue

        retro_product_ids = examples["retro_products"][i]
        retro_labels = [
            NO_LABEL_INDEX if label is None else label
            for label in examples["retro_labels"][i]
        ]
        properties = [
            NO_LABEL_INDEX if prop is None else prop for prop in examples["property"][i]
        ]

        input_ids, labels, molecule_ids, retro_product_ids, retro_labels = (
            encode_supervised_example(
                prompt=examples["prompt"][i],
                response=examples["response"][i],
                system=examples["system"][i],
                molecule_ids=examples["molecules"][i],
                retro_product_ids=retro_product_ids,
                retro_labels=retro_labels,
                template=template,
                tokenizer=tokenizer,
                data_args=data_args,
            )
        )
        # molecule_ids = examples["molecules"][i]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["molecule_ids"].append(molecule_ids)
        model_inputs["molecule_properties"].append(properties)
        model_inputs["retro_labels"].append(retro_labels)
        model_inputs["retro_product_ids"].append(retro_product_ids)

    return model_inputs

def print_supervised_dataset_example(
    example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer"
) -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("Print_supervised_dataset_example")

    print("input_ids:\n{}".format(example["input_ids"]))
    print(
        "inputs:\n{}".format(
            tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        )
    )
    print("label_ids:\n{}".format(example["labels"]))
    print(
        "labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False))
    )
    print("molecule_ids:\n{}".format(example["molecule_ids"]))
    print("molecule_properties:\n{}".format(example["molecule_properties"]))
    print("retro_labels:\n{}".format(example["retro_labels"]))
    print("retro_product_ids:\n{}".format(example["retro_product_ids"]))
