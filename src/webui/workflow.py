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

import re
import os
import json
import math
import torch
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, List, Optional, Dict, Any

from ..data import get_dataset, DataCollatorForSeqGraph, get_template_and_fix_tokenizer
from ..extras.constants import IGNORE_INDEX, NO_LABEL_INDEX
from ..extras.misc import get_logits_processor
from ..extras.ploting import plot_loss
from ..model import load_tokenizer, GraphLLMForCausalMLM
from ..hparams import get_train_args
from .dataset import MolQADataset

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from ..hparams import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
    )

def remove_extra_spaces(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()

def load_model_and_tokenizer(args):
    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(args)
    )
    tokenizer = load_tokenizer(model_args, generate_mode=True)["tokenizer"]
    tokenizer.pad_token = tokenizer.eos_token

    model = GraphLLMForCausalMLM.from_pretrained(
        tokenizer, model_args, data_args, training_args, finetuning_args, load_adapter=True
    )

    return model, tokenizer, generating_args

def process_input(input_data: Dict[str, Any], model, tokenizer, generating_args: "GeneratingArguments"):
    
    dataset = MolQADataset([input_data], tokenizer, generating_args.max_length)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False
    )

    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    return dataloader, gen_kwargs

def generate(model, dataloader, gen_kwargs):
    property_names = ["BBBP", "HIV", "BACE", "CO2", "N2", "O2", "FFV", "TC", "SC", "SA"]

    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        property_data = batch["property"].to(model.device)

        model.eval()
        with torch.no_grad():
            all_info_dict = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                molecule_properties=property_data,
                do_molecular_design=True,
                do_retrosynthesis=True,
                expansion_topk=50,
                iterations=100,
                max_planning_time=30,
                rollback=True,
                **gen_kwargs,
            )

            assert len(all_info_dict["smiles_list"]) == 1

            for i in range(len(all_info_dict["smiles_list"])):
                llm_response = "".join(item for item in all_info_dict["text_lists"][i] if item is not None)
                result = {
                    "llm_smiles": all_info_dict["smiles_list"][i],
                    "property": {},
                }
                for j, prop_name in enumerate(property_names):
                    prop_value = property_data[i][j].item()
                    if not math.isnan(prop_value):
                        result["property"][prop_name] = prop_value

                retro_plan = all_info_dict["retro_plan_dict"][result["llm_smiles"]]
                result["llm_reactions"] = []
                if retro_plan["success"]:
                    for reaction, template, cost in zip(
                        retro_plan["reaction_list"],
                        retro_plan["templates"],
                        retro_plan["cost"],
                    ):
                        result["llm_reactions"].append(
                            {"reaction": reaction, "template": template, "cost": cost}
                        )
                result["llm_response"] = remove_extra_spaces(llm_response)
                return result