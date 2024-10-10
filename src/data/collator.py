import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from torch_geometric.data import Batch as PyGBatch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """
    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class DataCollatorForSeqGraph:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    """
    tokenizer: PreTrainedTokenizerBase
    mol_id_to_pyg: Dict[str, Any]
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        if labels is not None and all(label is None for label in labels):
            labels = None

        # Store molecule_ids, retro_labels, and retro_product_ids separately and remove from non_labels_features
        molecule_ids_list = []
        retro_labels_list = []
        retro_products_list = []
        non_labels_features = []
        for feature in features:
            new_feature = {k: v for k, v in feature.items() if k != label_name}
            if 'molecule_ids' in new_feature:
                molecule_ids_list.append(new_feature['molecule_ids'])
                del new_feature['molecule_ids']
            else:
                molecule_ids_list.append(None)
            if 'retro_labels' in new_feature:
                retro_labels_list.append(new_feature['retro_labels'])
                del new_feature['retro_labels']
            else:
                retro_labels_list.append(None)
            if 'retro_product_ids' in new_feature:
                retro_products_list.append(new_feature['retro_product_ids'])
                del new_feature['retro_product_ids']
            else:
                retro_products_list.append(None)
            non_labels_features.append(new_feature)

        # Convert molecule IDs to PyG Data objects
        molecule_graphs_list = []
        design_graphs_list = []
        for seq_idx, molecule_ids in enumerate(molecule_ids_list):
            if molecule_ids is not None and len(molecule_ids) > 0:
                for pos, mol_id in enumerate(molecule_ids):
                    if pos == 0:
                        design_graphs_list.append(self.mol_id_to_pyg[mol_id])
                    if mol_id != self.label_pad_token_id and mol_id in self.mol_id_to_pyg:
                        molecule_graphs_list.append(self.mol_id_to_pyg[mol_id])

        # Convert retro_product_ids to PyG Data objects
        retro_product_graphs_list = []
        for seq_idx, retro_product_ids in enumerate(retro_products_list):
            if retro_product_ids is not None and len(retro_product_ids) > 0:
                for pos, mol_id in enumerate(retro_product_ids):
                    if mol_id != self.label_pad_token_id and mol_id in self.mol_id_to_pyg:
                        retro_product_graphs_list.append(self.mol_id_to_pyg[mol_id])

        # Batch the PyG Data objects
        if molecule_graphs_list:
            batched_graphs = PyGBatch.from_data_list(molecule_graphs_list)
        else:
            batched_graphs = None
        
        if design_graphs_list:
            batched_design_graphs = PyGBatch.from_data_list(design_graphs_list)
        else:
            batched_design_graphs = None

        if retro_product_graphs_list:
            batched_retro_products = PyGBatch.from_data_list(retro_product_graphs_list)
        else:
            batched_retro_products = None

        # Pad retro_labels
        if retro_labels_list and any(retro_labels is not None for retro_labels in retro_labels_list):
            max_retro_length = max(len(retro_labels) for retro_labels in retro_labels_list if retro_labels is not None)
            padded_retro_labels = [
                retro_labels + [self.label_pad_token_id] * (max_retro_length - len(retro_labels)) if retro_labels is not None else [self.label_pad_token_id] * max_retro_length
                for retro_labels in retro_labels_list
            ]
        else:
            padded_retro_labels = None

        # Pad other features
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        batch["molecule_graphs"] = batched_graphs
        batch["design_graphs"] = batched_design_graphs
        batch["retro_product_graphs"] = batched_retro_products
        batch["retro_labels"] = torch.tensor(padded_retro_labels, dtype=torch.int64)

        # Pad labels
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            padded_labels = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                if padding_side == "right"
                else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                for label in labels
            ]
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.int64)

        # Prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch