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

import inspect
import os
import sys
from typing import TYPE_CHECKING, Literal, Optional, Union
from functools import partial

import numpy as np
from datasets import load_dataset, load_from_disk

# from ..extras.constants import FILEEXT2TYPE
from ..extras.logging import get_logger
from ..extras.misc import has_tokenized_data
from .aligner import align_dataset
from .data_utils import merge_dataset
from .parser import get_dataset_attr
# from .preprocess import get_preprocess_and_print_func
from .template import get_template_and_fix_tokenizer

from .processors.mmsupervised import (
    preprocess_mmsupervised_dataset,
    print_supervised_dataset_example,
    encode_graph_pyg
)

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .parser import DatasetAttr


logger = get_logger(__name__)


def load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    logger.info("Loading dataset {}...".format(dataset_attr))

    data_files = []
    assert dataset_attr.load_from == "file"

    data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
    data_files.append(data_path)
    data_path = data_path.split(".")[-1]

    if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
        kwargs = {"trust_remote_code": True}
    else:
        kwargs = {}

    dataset = load_dataset(
        path=data_path,
        name=None,
        data_dir=None,
        data_files=data_files,
        split=data_args.split,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
        streaming=False,
        **kwargs,
    )
    
    converted_dataset, mol_id_to_smiles = align_dataset(dataset, dataset_attr, data_args, training_args)
    return converted_dataset, mol_id_to_smiles

def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: "PreTrainedTokenizer",
) -> Union["Dataset", "IterableDataset"]:
    
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template, data_args.tool_format)
    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")
    print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)

    # Load tokenized dataset
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            mol_id_to_pyg = encode_graph_pyg(data_path=data_args.tokenized_path)
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset = load_from_disk(data_args.tokenized_path)
            logger.info("Loaded tokenized dataset from {}.".format(data_args.tokenized_path))
            # print_function(next(iter(dataset)))
            data_iter = iter(dataset)
            print_function(next(data_iter))
            return mol_id_to_pyg, dataset
        
    # Load tokenized dataset
    with training_args.main_process_first(desc="load dataset"):
        # current only support one dataset
        dataset_attr = get_dataset_attr(data_args)
        dataset, mol_id_to_smiles = load_single_dataset(dataset_attr, model_args, data_args, training_args)

    with training_args.main_process_first(desc="pre-process dataset"):
        preprocess_func = partial(
            preprocess_mmsupervised_dataset,
            template=template,
            tokenizer=tokenizer,
            data_args=data_args,
        )

        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )
        
        dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)
        
        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset.save_to_disk(data_args.tokenized_path)
                mol_id_to_pyg = encode_graph_pyg(data_path=data_args.tokenized_path, mol_id_to_smiles=mol_id_to_smiles)
                logger.info("Tokenized dataset saved at {}.".format(data_args.tokenized_path))
                logger.info("Please restart the training with `tokenized_path: {}`.".format(data_args.tokenized_path))
            sys.exit(0)
        else:
            mol_id_to_pyg = encode_graph_pyg(mol_id_to_smiles=mol_id_to_smiles)

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Cannot find valid samples.")
                
        return mol_id_to_pyg, dataset
