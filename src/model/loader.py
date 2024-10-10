# Copyright 2024 the LlamaFactory team and the Llamole team.
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

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict
from pathlib import Path
import json
import pandas as pd
import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead
from huggingface_hub import hf_hub_download

from ..extras.logging import get_logger
from ..extras.misc import (
    count_parameters,
    skip_check_imports,
    try_download_model_from_ms,
)
from .adapter import init_adapter
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model

from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model

from .graph_decoder.diffusion_model import GraphDiT
from .graph_encoder.model import GraphCLIP
from .graph_predictor.model import GraphPredictor

if TYPE_CHECKING:
    from transformers import (
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        ProcessorMixin,
    )

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def download_from_hf(repo_id, filename, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

def load_tokenizer(model_args: "ModelArguments", generate_mode=False) -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer or a pre-saved tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)

    padding_size = 'left' if generate_mode else 'right'
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side=padding_size,
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side=padding_size,
            **init_kwargs,
        )

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info(
            "Add {} to special tokens.".format(",".join(model_args.new_special_tokens))
        )

        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning(
                "New tokens have been added, changed `resize_vocab` to True."
            )

    patch_tokenizer(tokenizer)

    if model_args.new_special_tokens is not None:
        token_id_dict = {}
        for elem in model_args.new_special_tokens:
            if isinstance(elem, str) and len(elem) != 0:
                elem_token_ids = tokenizer.encode(elem, add_special_tokens=False)
                token_id_dict[elem] = elem_token_ids
        logger.info(f"Dictionary of added tokens and their IDs: {token_id_dict}")

    return {"tokenizer": tokenizer, "processor": None}

def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_language_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None
    lazy_load = False

    # if model is None and not lazy_load:
    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(**init_kwargs)
    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)
        
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if (
                param.data.dtype == torch.float32
                and model_args.compute_dtype != torch.float32
            ):
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "lm trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "lm all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model

def load_graph_decoder(model_args: "ModelArguments", path: str, device: str):
    path = Path(path)
    config_path = path / "config.yaml"
    
    if not config_path.exists():
        logger.info(f"Config not found in {path}. Downloading from Hugging Face.")
        repo_id = "liuganghuggingface/Llamole-Pretrained-GraphDiT"
        config_path = download_from_hf(repo_id, "config.yaml", path)
        download_from_hf(repo_id, "data.meta.json", path)
        download_from_hf(repo_id, "model.pt", path)

    data_info_path = path / "data.meta.json"

    model = GraphDiT(
        model_config_path=config_path,
        data_info_path=data_info_path,
        model_dtype=model_args.compute_dtype,
    )
    model.init_model(path)
    if model_args.disable_graph_model_gradient:
        model.disable_grads()
    model.to(device)

    for param in model.parameters():
        if param.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
            param.data = param.data.to(model_args.compute_dtype)

    trainable_params, all_param = count_parameters(model)
    param_stats = "Graph DiT trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    )
    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            logger.info(
                f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}"
            )

    return model

def load_graph_predictor(model_args: "ModelArguments", path: str, device: str):
    path = Path(path)
    config_path = path / "config.json"
    
    if not config_path.exists():
        logger.info(f"Config not found in {path}. Downloading from Hugging Face.")
        repo_id = "liuganghuggingface/Llamole-Pretrained-GNNPredictor"
        config_path = download_from_hf(repo_id, "config.json", path)
        download_from_hf(repo_id, "model.pt", path)
        download_from_hf(repo_id, "cost_model.pt", path)
        download_from_hf(repo_id, "label_to_template.csv.gz", path)
        download_from_hf(repo_id, "available.csv.gz", path)

    with open(config_path, "r") as f:
        config = json.load(f)

    label_to_template_path = path / "label_to_template.csv.gz"
    label_to_template_df = pd.read_csv(label_to_template_path, compression='gzip')
    label_to_template = dict(zip(label_to_template_df['rule_label'], label_to_template_df['retro_templates']))

    available_path = path / "available.csv.gz"
    available = pd.read_csv(available_path, compression='gzip')

    model = GraphPredictor(
        num_layer=config["num_layer"],
        hidden_size=config["hidden_size"],
        drop_ratio=config["drop_ratio"],
        out_dim=config["num_task"],
        model_config=config,
        label_to_template=label_to_template,
        available=available,
    )
    
    model.init_model(path)
    model.init_neural_cost(path)
    
    if model_args.disable_graph_model_gradient:
        model.disable_grads()
    
    model.to(device)

    for param in model.parameters():
        if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
            param.data = param.data.to(model_args.compute_dtype)

    trainable_params, all_param = count_parameters(model)
    param_stats = "Graph Predictor trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    )
    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            logger.info(
                f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}"
            )
            
    return model

def load_graph_encoder(model_args: "ModelArguments", path: str, device: str):
    path = Path(path)
    config_path = path / "config.json"
    
    if not config_path.exists():
        logger.info(f"Config not found in {path}. Downloading from Hugging Face.")
        repo_id = "liuganghuggingface/Llamole-Pretrained-GraphEncoder"
        config_path = download_from_hf(repo_id, "config.json", path)
        download_from_hf(repo_id, "model.pt", path)
        download_from_hf(repo_id, "model_proj.pt", path)

    with open(config_path, "r") as f:
        config = json.load(f)

    model = GraphCLIP(
        graph_num_layer=config["num_layer"],
        graph_hidden_size=config["hidden_size"],
        dropout=config["drop_ratio"],
        model_config=config,
    )
    model.init_model(path, verbose=False)
    if model_args.disable_graph_model_gradient:
        model.disable_grads()
    model.to(device)

    for param in model.parameters():
        if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
            param.data = param.data.to(model_args.compute_dtype)

    trainable_params, all_param = count_parameters(model)
    param_stats = "Graph CLIP Encoder trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    )

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            logger.info(
                f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}"
            )

    return model