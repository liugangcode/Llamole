# Copyright 2024 the Llamole Team
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
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils import ModelOutput
from transformers.generation.utils import LogitsProcessorList, GenerationConfig
from huggingface_hub import snapshot_download

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import os
import json
import time
from dataclasses import dataclass

from typing import Union, Tuple, Optional
from .loader import load_language_model, load_tokenizer
from .loader import load_graph_decoder, load_graph_predictor, load_graph_encoder
from ..extras.constants import NO_LABEL_INDEX, IGNORE_INDEX, BOND_INDEX

from .planner import molstar
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.utils import remove_isolated_nodes

# Save configuration
def convert_to_dict(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [convert_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {
            k: convert_to_dict(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
    else:
        return str(obj)  # Convert any other objects to string

@dataclass
class GraphLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    additional_log_info: Optional[Dict[str, float]] = None


class GraphLLMForCausalMLM(PreTrainedModel):
    def __init__(
        self,
        model_args,
        finetuning_args,
        data_args,
        language_model,
        graph_decoder,
        graph_predictor,
        graph_encoder,
        token_id_dict,
        tokenizer,
    ):
        super().__init__(language_model.config)
        self.language_model = language_model
        self.graph_decoder = graph_decoder
        self.graph_predictor = graph_predictor
        self.graph_encoder = graph_encoder
        
        self.token_id_dict = token_id_dict
        self.num_body_tokens = data_args.learned_query_size

        self.loss_weight_lm = finetuning_args.loss_weight_lm
        self.loss_weight_design = finetuning_args.loss_weight_design
        self.loss_weight_retro = finetuning_args.loss_weight_retro

        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.data_args = data_args
        self.tokenizer = tokenizer

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        model_args,
        data_args,
        training_args,
        finetuning_args,
        load_adapter=False,
        add_valuehead=False,
    ):
        if load_adapter:
            if model_args.adapter_name_or_path is None:
                raise ValueError("Please specify the adapter_name_or_path when load_adapter is True.")
            
            if len(model_args.adapter_name_or_path) != 1:
                raise ValueError("Only one adapter is supported at a time.")
            
            adapter_path = model_args.adapter_name_or_path[0]
            
            if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                # Download from HuggingFace
                adapter_name = os.path.basename(adapter_path)
                valid_adapters = [
                    "Llama-3.1-8B-Instruct-Adapter",
                    "Qwen2-7B-Instruct-Adapter",
                    "Mistral-7B-Instruct-v0.3-Adapter"
                ]
                
                if adapter_name not in valid_adapters:
                    raise ValueError(f"Invalid adapter name. Supported adapters are: {', '.join(valid_adapters)}")
                
                repo_id = f"liuganghuggingface/Llamole-{adapter_name}"
                print(f"Downloading adapter {adapter_name} from HuggingFace repo: {repo_id}")
                
                try:
                    # Download all files including subfolders to the adapter_path
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=adapter_path,
                        local_dir_use_symlinks=False,
                        ignore_patterns=["*.md", "*.txt"]  # Optionally ignore certain file types
                    )

                    print(f"Successfully downloaded all adapter files to {adapter_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download adapter files: {str(e)}")


        language_model = load_language_model(
            tokenizer,
            model_args,
            finetuning_args,
            training_args.do_train,
            add_valuehead,
        )

        device = next(language_model.parameters()).device

        graph_decoder = load_graph_decoder(
            model_args,
            path=model_args.graph_decoder_path,
            device=device,
        )

        graph_predictor = load_graph_predictor(
            model_args,
            path=model_args.graph_predictor_path,
            device=device,
        )

        graph_encoder = load_graph_encoder(
            model_args,
            path=model_args.graph_encoder_path,
            device=device,
        )

        if (
            getattr(language_model, "is_quantized", False)
            and not training_args.do_train
        ):
            setattr(
                language_model, "_hf_peft_config_loaded", True
            )  # hack here: make model compatible with prediction

        token_id_dict = {}
        for elem in model_args.new_special_tokens:
            if isinstance(elem, str) and len(elem) != 0:
                elem_token_ids = tokenizer.encode(elem, add_special_tokens=False)
                token_id_dict[elem] = elem_token_ids[0]

        model = cls(
            model_args=model_args,
            finetuning_args=finetuning_args,
            data_args=data_args,
            language_model=language_model,
            graph_decoder=graph_decoder,
            graph_predictor=graph_predictor,
            graph_encoder=graph_encoder,
            token_id_dict=token_id_dict,
            tokenizer=tokenizer,
        )

        graph_to_lm_connector = nn.Sequential(
            nn.Linear(graph_encoder.hidden_size, language_model.config.hidden_size),
            nn.SiLU(),
        )

        # Language Model to Graph Decoder connector
        lm_to_graph_decoder = nn.Sequential(
            nn.Linear(language_model.config.hidden_size, graph_decoder.text_input_size),
            nn.SiLU(),
        )

        # Language Model to Graph Predictor connector
        lm_to_graph_predictor = nn.Sequential(
            nn.Linear(
                language_model.config.hidden_size, graph_predictor.text_input_size
            ),
            nn.SiLU(),
        )

        for param in graph_to_lm_connector.parameters():
            if (
                param.dtype == torch.float32
                and model_args.compute_dtype != torch.float32
            ):
                param.data = param.data.to(model_args.compute_dtype)

        for param in lm_to_graph_decoder.parameters():
            if (
                param.dtype == torch.float32
                and model_args.compute_dtype != torch.float32
            ):
                param.data = param.data.to(model_args.compute_dtype)

        for param in lm_to_graph_predictor.parameters():
            if (
                param.dtype == torch.float32
                and model_args.compute_dtype != torch.float32
            ):
                param.data = param.data.to(model_args.compute_dtype)

        # Check if connector path is provided and load if available
        if load_adapter:
            if (
                hasattr(model_args, "graph_lm_connector_path")
                and model_args.graph_lm_connector_path
            ):
                connector_path = model_args.graph_lm_connector_path

                graph_to_lm_connector.load_state_dict(
                    torch.load(
                        os.path.join(connector_path, "graph_to_lm_connector.pt"),
                        map_location=device,
                        weights_only=True,
                    )
                )

                lm_to_graph_decoder.load_state_dict(
                    torch.load(
                        os.path.join(connector_path, "lm_to_graph_decoder.pt"),
                        map_location=device,
                        weights_only=True,
                    )
                )

                lm_to_graph_predictor.load_state_dict(
                    torch.load(
                        os.path.join(connector_path, "lm_to_graph_predictor.pt"),
                        map_location=device,
                        weights_only=True,
                    )
                )
            else:
                raise ValueError(f"Connector should be automatically downloaded with the adapter. Please manually download to the path {connector_path}")

        model.graph_to_lm_connector = graph_to_lm_connector
        model.lm_to_graph_decoder = lm_to_graph_decoder
        model.lm_to_graph_predictor = lm_to_graph_predictor
        model.graph_to_lm_connector.to(device)
        model.lm_to_graph_decoder.to(device)
        model.lm_to_graph_predictor.to(device)

        return model

    def to(self, device):
        super().to(device)
        self.language_model.to(device)
        self.graph_decoder.to(device)
        self.graph_predictor.to(device)
        self.graph_encoder.to(device)
        self.graph_to_lm_connector.to(device)
        self.lm_to_graph_decoder.to(device)
        self.lm_to_graph_predictor.to(device)
        return self

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        molecule_graphs: Optional[PyGBatch] = None,
        molecule_properties: Optional[torch.FloatTensor] = None,
        design_graphs: Optional[PyGBatch] = None,
        retro_labels: Optional[torch.LongTensor] = None,
        retro_product_graphs: Optional[PyGBatch] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, GraphLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        mol_token_id = self.token_id_dict["<molecule>"]
        design_start_token_id = self.token_id_dict["<design_start>"]
        retro_start_token_id = self.token_id_dict["<retro_start>"]

        # PeftModelForCausalLM -> LlamaForCausalLM -> LlamaModel
        base_llm = self.language_model.model.model
        inputs_embeds = base_llm.embed_tokens(input_ids)
        mol_positions = (input_ids == mol_token_id).nonzero()

        mol_embeds = self.graph_encoder(
            molecule_graphs.x,
            molecule_graphs.edge_index,
            molecule_graphs.edge_attr,
            molecule_graphs.batch,
        )
        mol_embeds = self.graph_to_lm_connector(mol_embeds)

        assert (
            mol_positions.shape[0] == mol_embeds.shape[0]
        ), f"Number of molecule tokens ({mol_positions.shape[0]}) does not match number of molecule embeddings ({mol_embeds.shape[0]})"

        inputs_embeds[mol_positions[:, 0], mol_positions[:, 1]] = mol_embeds.to(
            inputs_embeds.dtype
        )

        lm_outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )

        lm_loss = lm_outputs.loss
        lm_hidden_states = lm_outputs.hidden_states[-1]

        design_loss = 0
        if design_graphs is not None:
            design_start_positions = (input_ids == design_start_token_id).nonzero()
            design_body_start = design_start_positions[:, 1] + 1
            design_body_indices = design_body_start.unsqueeze(1) + torch.arange(
                self.num_body_tokens, device=input_ids.device
            )
            design_hidden = lm_hidden_states[
                design_start_positions[:, 0].unsqueeze(1), design_body_indices[:, 1]
            ].mean(dim=1)
            if design_start_positions.numel() > 0:
                design_hidden = self.lm_to_graph_decoder(design_hidden)
                design_loss = self.graph_decoder(
                    design_graphs.x,
                    design_graphs.edge_index,
                    design_graphs.edge_attr,
                    design_graphs.batch,
                    molecule_properties,
                    design_hidden,
                    NO_LABEL_INDEX,
                )

        # Process retro labels
        retro_loss = 0
        if retro_labels is not None:
            # Get retro start positions for valid retro labels: (batch, step)
            retro_start_positions = (input_ids == retro_start_token_id).nonzero()
            retro_labels = retro_labels[retro_labels != IGNORE_INDEX]
            valid_retro_mask = retro_labels != NO_LABEL_INDEX
            retro_start_positions = retro_start_positions[valid_retro_mask]
            retro_labels = retro_labels[valid_retro_mask]

            if len(retro_labels) > 0:
                # Get the query hidden states for each retro prediction
                retro_body_start = retro_start_positions[:, 1] + 1
                retro_body_indices = retro_body_start.unsqueeze(1) + torch.arange(
                    self.num_body_tokens, device=input_ids.device
                )
                retro_hidden = lm_hidden_states[
                    retro_start_positions[:, 0].unsqueeze(1), retro_body_indices
                ].mean(dim=1)

                # Prepare graph inputs
                retro_product_graphs = retro_product_graphs[
                    valid_retro_mask.nonzero().view(-1)
                ]
                retro_product_graphs = PyGBatch.from_data_list(retro_product_graphs)

                # Transform hidden states and make predictions
                retro_hidden = self.lm_to_graph_predictor(retro_hidden)
                retro_pred = self.graph_predictor(
                    retro_product_graphs.x,
                    retro_product_graphs.edge_index,
                    retro_product_graphs.edge_attr,
                    retro_product_graphs.batch,
                    retro_hidden,
                )
                retro_loss = F.cross_entropy(
                    retro_pred,
                    retro_labels,
                )

        total_loss = (
            self.loss_weight_lm * lm_loss
            + self.loss_weight_design * retro_loss
            + self.loss_weight_retro * retro_loss
        )

        if not return_dict:
            output = (lm_outputs.logits,) + lm_outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GraphLMOutput(
            loss=total_loss,
            logits=lm_outputs.logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        save_graph_modules: bool = False,
        **kwargs,
    ):
        """
        Save the model and its configuration file to a directory.
        """
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        # Save language model
        language_model_path = os.path.join(save_directory)
        self.language_model.save_pretrained(
            language_model_path,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=False,  # set to false
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
        )

        if save_graph_modules:
            # Save graph models
            graph_models = {
                "graph_decoder": self.graph_decoder,
                "graph_predictor": self.graph_predictor,
                "graph_encoder": self.graph_encoder,
            }
            for name, model in graph_models.items():
                model_path = os.path.join(save_directory, name)
                model.save_pretrained(model_path)

        # Save additional components
        additional_components = {
            "graph_to_lm_connector": self.graph_to_lm_connector,
            "lm_to_graph_decoder": self.lm_to_graph_decoder,
            "lm_to_graph_predictor": self.lm_to_graph_predictor,
        }
        connector_path = os.path.join(save_directory, "connector")
        for name, component in additional_components.items():
            os.makedirs(connector_path, exist_ok=True)
            component_path = os.path.join(connector_path, f"{name}.pt")
            torch.save(component.state_dict(), component_path)

        config_dict = {
            "model_args": convert_to_dict(self.model_args),
            "finetuning_args": convert_to_dict(self.finetuning_args),
            "data_args": convert_to_dict(self.data_args),
            "token_id_dict": self.token_id_dict,
            "num_body_tokens": self.num_body_tokens,
            "loss_weight_lm": self.loss_weight_lm,
            "loss_weight_design": self.loss_weight_design,
            "loss_weight_retro": self.loss_weight_retro,
        }

        config_path = os.path.join(save_directory, "graphllm_config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Push to hub if required
        if push_to_hub:
            raise NotImplementedError("Push to hub not implemented yet")

    def add_special_body_tokens(
        self,
        input_ids: torch.LongTensor,
        body_token_id: int,
        num_body_tokens: int,
        start_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        batch_size, seq_length = input_ids.shape
        start_len = 1 if start_token_id is not None else 0
        if seq_length < num_body_tokens + start_len:
            seq_length = seq_length + num_body_tokens + start_len

        # Create a tensor to hold start positions for each batch item
        start_positions = torch.full(
            (batch_size,),
            seq_length - start_len - num_body_tokens,
            device=input_ids.device,
        )
        # Calculate how many tokens to keep from the original input
        tokens_to_keep = seq_length - num_body_tokens

        # Find start positions
        if start_token_id is not None:
            start_pos_rows, start_pos_cols = (input_ids == start_token_id).nonzero(
                as_tuple=True
            )
            for row, col in zip(start_pos_rows, start_pos_cols):
                start_positions[row] = col
            tokens_to_keep = seq_length - num_body_tokens - 1

        # Create body tokens
        body_tokens = torch.full(
            (batch_size, num_body_tokens), body_token_id, device=input_ids.device
        )

        # Create new input_ids with left padding
        new_input_ids = torch.full(
            (batch_size, seq_length),
            self.tokenizer.eos_token_id,
            device=input_ids.device,
        )

        for i in range(batch_size):
            start_pos = start_positions[i]
            # Keep the rightmost tokens_to_keep tokens before the start token
            keep_start = max(0, start_pos - tokens_to_keep)

            if start_token_id is not None:
                new_input_ids[
                    i, -(num_body_tokens + 1 + (start_pos - keep_start)) :
                ] = torch.cat(
                    [
                        input_ids[i, keep_start:start_pos],
                        torch.LongTensor([start_token_id]).to(input_ids.device),
                        body_tokens[i],
                    ]
                )
            else:
                new_input_ids[
                    i, -(num_body_tokens + 1 + (start_pos - keep_start)) :
                ] = torch.cat([input_ids[i, keep_start:start_pos], body_tokens[i]])
        return new_input_ids

    @torch.no_grad()
    def design_molecule(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        molecule_properties: Optional[torch.FloatTensor] = None,
        molecule_graphs: Optional[PyGBatch] = None,
        rollback: bool = False,
        **kwargs,
    ) -> List[Optional[str]]:
        design_start_token_id = self.token_id_dict["<design_start>"]
        design_body_token_id = self.token_id_dict["<design_body>"]

        # 1. Generate molecular design analysis
        if molecule_graphs is None:
            analysis_tokens = self.language_model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            analysis_tokens = analysis_tokens[:, input_ids.shape[1] :]
        else:
            mol_token_id = self.token_id_dict["<molecule>"]
            base_llm = self.language_model.model
            inputs_embeds = base_llm.embed_tokens(input_ids)

            mol_positions = (input_ids == mol_token_id).nonzero()
            mol_embeds = self.graph_encoder(
                molecule_graphs.x,
                molecule_graphs.edge_index,
                molecule_graphs.edge_attr,
                molecule_graphs.batch,
            )
            mol_embeds = self.graph_to_lm_connector(mol_embeds)

            assert (
                mol_positions.shape[0] == mol_embeds.shape[0]
            ), f"Number of molecule tokens ({mol_positions.shape[0]}) does not match number of molecule embeddings ({mol_embeds.shape[0]})"
            inputs_embeds[mol_positions[:, 0], mol_positions[:, 1]] = mol_embeds.to(
                inputs_embeds.dtype
            )
            analysis_tokens = self.language_model.generate(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            ) # no input

        # 2. Add special tokens for design body
        design_input_ids = self.add_special_body_tokens(
            analysis_tokens,
            design_body_token_id,
            self.num_body_tokens,
            start_token_id=design_start_token_id,
        )
        design_input_ids = torch.cat([input_ids, design_input_ids], dim=1)

        # 3. Get LLM embeddings for design body
        lm_outputs = self.language_model(
            input_ids=design_input_ids,
            attention_mask=torch.ones_like(design_input_ids),
            output_hidden_states=True,
            return_dict=True,
        )
        lm_hidden_states = lm_outputs.hidden_states[-1]
        design_hidden = lm_hidden_states[:, -self.num_body_tokens :].mean(dim=1)

        # 4. Generate molecules using graph decoder
        design_hidden = self.lm_to_graph_decoder(design_hidden)
        molecule_properties = molecule_properties.type_as(design_hidden)
        smiles_list = self.graph_decoder.generate(
            molecule_properties,
            design_hidden,
            NO_LABEL_INDEX,
        )

        # Handle None values in smiles_list
        if rollback and None in smiles_list:
            smiles_list = self.design_rollback(design_input_ids, smiles_list, **kwargs)

        return analysis_tokens, smiles_list

    def design_rollback(
        self,
        analysis_tokens: torch.LongTensor,
        smiles_list: List[Optional[str]],
        **kwargs,
    ) -> List[Optional[str]]:
        rollback_token_id = self.token_id_dict.get("<rollback_start>")
        rollback_end_token_id = self.token_id_dict.get("<rollback_end>")
        none_indices = [i for i, smiles in enumerate(smiles_list) if smiles is None]

        if not none_indices:
            return smiles_list  # No None values, return original list

        # Get corresponding analysis tokens for None indices
        none_indices = torch.LongTensor(none_indices)
        rollback_analysis_tokens = analysis_tokens[none_indices]

        # Add rollback token to the end of each analysis token sequence
        rollback_input_ids = self.add_special_body_tokens(
            rollback_analysis_tokens,
            rollback_token_id,
            1,
        )

        if "max_new_tokens" in kwargs:
            kwargs["max_new_tokens"] *= 2

        # Generate new tokens
        new_tokens = self.language_model.generate(
            inputs=rollback_input_ids,
            attention_mask=torch.ones_like(rollback_input_ids),
            **kwargs,
        )

        # Process and decode new tokens
        new_smiles = []
        for seq in new_tokens[:, rollback_input_ids.shape[1] :]:
            decoded_seq = self.tokenizer.decode(seq, skip_special_tokens=False)
            end_smiles_pos = decoded_seq.find(
                self.tokenizer.decode([rollback_end_token_id])
            )

            if end_smiles_pos != -1:
                # If end token is found, truncate the sequence
                new_smiles.append(decoded_seq[:end_smiles_pos].strip())
            else:
                # If end token is not found, append None
                new_smiles.append(None)

        # Update smiles_list with new decoded tokens
        for i, new_smiles_str in zip(none_indices, new_smiles):
            smiles_list[i] = new_smiles_str

        return smiles_list

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES string: {smiles}")
            return None

        type_idx = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:  # Exclude hydrogen atoms
                type_idx.append(
                    119 - 2 if atom.GetSymbol() == "*" else atom.GetAtomicNum() - 2
                )

        x = torch.LongTensor(type_idx)
        num_nodes = x.size(0)

        # Initialize edge_index and edge_attr as empty tensors
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.long)

        # Only process bonds if they exist
        if mol.GetNumBonds() > 0:
            bond_src = []
            bond_dst = []
            bond_type = []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                # Exclude bonds involving hydrogen atoms
                if mol.GetAtomWithIdx(start).GetAtomicNum() != 1 and mol.GetAtomWithIdx(end).GetAtomicNum() != 1:
                    bond_src.extend([start, end])
                    bond_dst.extend([end, start])
                    bond_type.extend([BOND_INDEX.get(bond.GetBondType(), 1)] * 2)

            if bond_src:  # Only create edge_index and edge_attr if there are valid bonds
                edge_index = torch.tensor([bond_src, bond_dst], dtype=torch.long)
                edge_attr = torch.tensor(bond_type, dtype=torch.long)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        
        return data

    def retrosynthesize_rollback(self, input_ids, design_text, smiles, **kwargs):
        input_text = f"{design_text} To synthesize {smiles}, follow these procedures: "
        input_tokens = self.tokenizer.encode(
            input_text, add_special_tokens=False, return_tensors="pt"
        )
        input_tokens = input_tokens.to(self.device)

        if "max_new_tokens" in kwargs:
            kwargs["max_new_tokens"] = 256

        # Generate tokens
        generated_tokens = self.language_model.generate(
            inputs=input_tokens,
            **kwargs,
        )
        generated_tokens = generated_tokens[:, input_tokens.shape[1] :]
        generated_tokens = generated_tokens.cpu().squeeze().tolist()
        new_input_text = f"To synthesize {smiles}, follow these procedures: "
        new_input_tokens = self.tokenizer.encode(new_input_text)
        generated_tokens = new_input_tokens + generated_tokens
        return generated_tokens
    
    def one_step_reaction(
        self,
        product_smiles,
        input_ids,
        design_text,
        molecule_graphs,
        topk,
        **kwargs,
    ):  
        # 1. Generate retrosynthesis analysis
        retro_start_token_id = self.token_id_dict["<retro_start>"]
        retro_body_token_id = self.token_id_dict["<retro_body>"]
        mol_token_id = self.token_id_dict["<molecule>"]

        input_text = f"{design_text} To synthesize <molecule>, follow these procedures: "
        
        prompt_tokens = self.tokenizer.encode(
            input_text, add_special_tokens=False, return_tensors="pt"
        )
        prompt_tokens = prompt_tokens.to(self.device)

        # Combine input_ids with new_prompt_tokens if input_ids is provided
        if input_ids is not None and molecule_graphs is not None:
            input_ids = input_ids.view(1, -1)
            prompt_tokens = torch.cat([input_ids, prompt_tokens], dim=-1)
        
        base_llm = self.language_model.model
        inputs_embeds = base_llm.embed_tokens(prompt_tokens)

        product_graph = self.smiles_to_graph(product_smiles)
        if product_graph is None:
            return {
                "reactants": [],
                "scores": [],
                "templates": [],
                "analysis": self.tokenizer.encode(
                    "Invalid product SMILES", add_special_tokens=False
                ),
            }
        product_graph.to(self.device)

        if input_ids is not None and molecule_graphs is not None:
            all_graphs = PyGBatch.from_data_list(molecule_graphs.to_data_list() + [product_graph])
        else:
            all_graphs = PyGBatch.from_data_list([product_graph])
        mol_embeds = self.graph_encoder(
            all_graphs.x,
            all_graphs.edge_index,
            all_graphs.edge_attr,
            all_graphs.batch,
        )
        mol_embeds = self.graph_to_lm_connector(mol_embeds)

        mol_positions = (prompt_tokens == mol_token_id).nonzero()
        assert (
            mol_positions.shape[0] == mol_embeds.shape[0]
        ), f"Number of molecule tokens ({mol_positions.shape[0]}) does not match number of molecule embeddings ({mol_embeds.shape[0]})"
        inputs_embeds[mol_positions[:, 0], mol_positions[:, 1]] = mol_embeds.to(
            inputs_embeds.dtype
        )
        attention_mask = torch.ones_like(prompt_tokens)

        if "max_new_tokens" in kwargs:
            kwargs["max_new_tokens"] = 512

        analysis_tokens = self.language_model.generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # 2. Encode analysis with query tokens
        retro_input_ids = self.add_special_body_tokens(
            analysis_tokens,
            retro_body_token_id,
            self.num_body_tokens,
            start_token_id=retro_start_token_id,
        )
        # Get LLM embeddings for retro body
        lm_outputs = self.language_model(
            input_ids=retro_input_ids,
            attention_mask=torch.ones_like(retro_input_ids),
            output_hidden_states=True,
            return_dict=True,
        )
        lm_hidden_states = lm_outputs.hidden_states[-1]
        retro_hidden = lm_hidden_states[:, -self.num_body_tokens :].mean(dim=1)
        retro_hidden = self.lm_to_graph_predictor(retro_hidden)

        # 3. Sample retrosynthetic templates
        reactants, scores, templates = self.graph_predictor.sample_templates(
            product_graph, retro_hidden, product_smiles, topk
        )

        # 4. Adjust the input part from the generated tokens
        analysis_tokens = analysis_tokens.cpu().squeeze().tolist()
        input_text = f"To synthesize {product_smiles}, follow these procedures: "
        new_input_tokens = self.tokenizer.encode(input_text)
        analysis_tokens = new_input_tokens + analysis_tokens

        return {
            "reactants": reactants,
            "scores": scores,
            "templates": templates,
            "analysis": analysis_tokens,
        }

    @torch.no_grad()
    def estimate_synthesis_complexity(
        self,
        smiles: str,
        input_ids=None,
        reaction=None,
        molecule_cost_weight: float = 0,
        language_cost_weight: float = 1,
        reference_tokens: Optional[torch.LongTensor] = None,
    ):
        cost = 0

        if molecule_cost_weight is not None and molecule_cost_weight > 0:
            mol_cost = self.graph_predictor.estimate_cost(smiles)
            cost += mol_cost * molecule_cost_weight

        if language_cost_weight is not None and language_cost_weight > 0:
            language_cost = 0
            if reaction is None:
                message_content = f"""
                Estimate remaining steps for the target {smiles} consider the following factors::
                1. Intermediate complexity
                2. Reagent availability
                3. Side reactions
                4. Stereochemistry challenges"""
            else:
                step = reaction.depth + 1
                template = reaction.template
                # analysis_tokens = reaction.analysis_tokens
                reactants = reaction.children
                reactants = ", ".join([r.mol for r in reactants])
                message_content = f"""
                Estimate remaining steps for the target {smiles} given the following parameters:
                Current step {step},
                Current template: {template},
                Reactants: {reactants}. 
                Consider the following factors:
                1. Intermediate complexity
                2. Reagent availability
                3. Side reactions
                4. Stereochemistry challenges"""

            # Create the messages list for the chat template
            messages = [{"role": "user", "content": message_content}]

            # Apply the chat template
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            answers = [
                "All readily available",
                "Some commercial, some need 1-2 steps",
                "Mix of commercial and multi-step synthesis",
                "Mostly require complex synthesis",
                "All require extensive multi-step synthesis",
            ]

            answer_costs = [0, 1, 2.5, 4.5, 7]
            answer_messages = [
                [
                    {
                        "role": "user",
                        "content": "Estimate the synthesis complexity:",
                    },
                    {"role": "assistant", "content": answer},
                ]
                for answer in answers
            ]
            answer_chat_texts = [
                self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                for msg in answer_messages
            ]

            # Encode chat texts
            input_ids = self.tokenizer.encode(chat_text, return_tensors="pt").to(
                self.device
            )
            answer_tokens = [
                self.tokenizer.encode(text) for text in answer_chat_texts
            ]

            # Get logits from the language model
            outputs = self.language_model(input_ids)
            logits = outputs.logits[:, -1, :]

            # Calculate softmax probabilities for each answer
            answer_logits = torch.stack(
                [logits[:, tokens].mean(dim=1) for tokens in answer_tokens]
            )
            probs = torch.nn.functional.softmax(answer_logits, dim=0)
            language_cost = (
                (probs * torch.tensor(answer_costs, device=probs.device))
                .sum()
                .item()
            )

            language_cost = language_cost * language_cost_weight
            cost += language_cost

        return cost

    @torch.no_grad()
    def retrosynthesize(
        self,
        input_ids: torch.LongTensor,
        smiles: Optional[str] = None,
        molecule_graphs: Optional[PyGBatch] = None,
        expansion_topk: int = 50,
        iterations: int = 100,
        starting_mols: Optional[List[str]] = None,
        molecule_cost_weight: float = 0,
        language_cost_weight: float = 1,
        max_planning_time: int = 300,
        rollback: bool = True,
        design_text: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Initialize variables
        target_smiles = None
        success = False
        reaction_list = None
        template_list = None
        analysis_tokens_list = None
        route_length = None
        total_time = 0.0
        cost = None

        # Handle starting molecules
        if starting_mols is None:
            if self.graph_predictor.available is None:
                raise ValueError(
                    "No starting molecules provided and no available starting molecules found."
                )
            starting_mols = self.graph_predictor.available["smiles"].tolist()

        # Handle case when no SMILES is provided
        if smiles is None and rollback:
            generated_tokens = self.retrosynthesize_rollback(input_ids, design_text, None, **kwargs)
            return self._create_failure_result(None, generated_tokens)

        # Preprocess SMILES
        target_smiles = smiles.replace("*", "[H]") if "*" in smiles else smiles

        # Check validity and handle rollback if necessary
        if not self.graph_decoder.check_valid(target_smiles) and rollback:
            generated_tokens = self.retrosynthesize_rollback(
                input_ids, design_text, target_smiles, **kwargs
            )
            return self._create_failure_result(target_smiles, generated_tokens)

        # Perform retrosynthesis
        t0 = time.time()

        def expand_fn(s):
            return self.one_step_reaction(
                s, input_ids=input_ids, design_text=design_text, molecule_graphs=molecule_graphs, topk=expansion_topk, **kwargs
            )

        def value_fn(s, r):
            return self.estimate_synthesis_complexity(
                s, input_ids, r, molecule_cost_weight, language_cost_weight
            )

        if target_smiles is None:
            return self._create_failure_result(None)

        success, best_route, iterations = molstar(
            target_mol=target_smiles,
            target_mol_id=0,
            starting_mols=starting_mols,
            expand_fn=expand_fn,
            value_fn=value_fn,
            iterations=iterations,
            max_time=max_planning_time,
        )

        total_time = time.time() - t0

        # Handle successful retrosynthesis
        if success:
            reaction_list, template_list, cost, analysis_tokens_list = best_route.get_reaction_list()
            route_length = best_route.length
        # Handle failed retrosynthesis with rollback
        elif rollback:
            generated_tokens = self.retrosynthesize_rollback(
                input_ids, design_text, target_smiles, **kwargs
            )
            return self._create_failure_result(target_smiles, generated_tokens)

        # Prepare and return result
        return {
            "target": target_smiles,
            "success": success,
            "time": total_time,
            "reaction_list": reaction_list,
            "cost": cost,
            "templates": template_list,
            "analysis_tokens": analysis_tokens_list,
            "route_length": route_length,
        }

    def _create_failure_result(
        self,
        target_smiles: Optional[str],
        generated_tokens: Optional[Union[torch.Tensor, list]] = None,
    ) -> Dict[str, Any]:
        return {
            "target": target_smiles,
            "success": False,
            "time": 0.0,
            "reaction_list": None,
            "cost": None,
            "templates": None,
            "analysis_tokens": (
                generated_tokens
                if generated_tokens is not None
                else "<NO ANALYSIS>"
            ),
            "route_length": None,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        molecule_properties: Optional[torch.FloatTensor] = None,
        molecule_graphs: Optional[PyGBatch] = None,
        rollback: bool = False,
        starting_mols: Optional[List[str]] = None,
        expansion_topk: int = 50,
        iterations: int = 100,
        molecule_cost_weight: float = 0,
        language_cost_weight: float = 1,
        do_molecular_design: Optional[bool] = True,
        do_retrosynthesis: bool = True,
        input_smiles_list: Optional[List[str]] = None,
        max_planning_time: int = 30,
        design_text_list: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        all_info_dict = {
            "token_lists": [],
            "text_lists": [],
            "design_analysis_tokens": None,
            "smiles_list": None,
            "retro_plan_dict": None,
        }

        # Molecular design
        if do_molecular_design is True:
            design_analysis_tokens, smiles_list = self.design_molecule(
                input_ids,
                attention_mask,
                molecule_properties,
                molecule_graphs,
                rollback,
                **kwargs,
            )
            all_info_dict["design_analysis_tokens"] = design_analysis_tokens.cpu()
            all_info_dict["smiles_list"] = smiles_list
        elif input_smiles_list is not None:
            all_info_dict["smiles_list"] = input_smiles_list
        else:
            raise ValueError(
                "Either do_molecular_design must be True/False or input_smiles_list must be provided."
            )
        
        # Retrosynthesis
        if do_retrosynthesis:
            if all_info_dict["smiles_list"] is None:
                raise ValueError(
                    "Either molecular design must be performed or input_smiles_list must be provided for retrosynthesis."
                )

            all_info_dict["retro_plan_dict"] = {}
            for i, smiles in enumerate(all_info_dict["smiles_list"]):
                if design_text_list is not None:
                    design_text = design_text_list[0]
                else:
                    design_text = None
                all_info_dict["retro_plan_dict"][smiles] = self.retrosynthesize(
                    input_ids[i] if input_ids.dim() > 1 else input_ids,
                    smiles,
                    molecule_graphs=molecule_graphs,
                    starting_mols=starting_mols,
                    expansion_topk=expansion_topk,
                    iterations=iterations,
                    molecule_cost_weight=molecule_cost_weight,
                    language_cost_weight=language_cost_weight,
                    max_planning_time=max_planning_time,
                    design_text=design_text,
                    **kwargs,
                )
        else:
            all_info_dict["retro_plan_dict"] = {
                smile: {"success": None} for smile in all_info_dict["smiles_list"]
            }

        for batch_idx, generated_mol in enumerate(all_info_dict["smiles_list"]):
            token_list = []
            text_list = []
            ignore_positions = {}
            if do_molecular_design:
                design_tokens = all_info_dict["design_analysis_tokens"][
                    batch_idx
                ].tolist()
                token_list = design_tokens + [IGNORE_INDEX]
                if generated_mol is None:
                    generated_mol = "<NO MOLECULE>"
                text_list = [
                    self.tokenizer.decode(
                        design_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaced=True,
                    ),
                    generated_mol + ". ",
                ]

                ignore_positions = {0: generated_mol}

            if do_retrosynthesis:
                available_mols = self.graph_predictor.available["smiles"].tolist()
                retro_plan = all_info_dict["retro_plan_dict"][generated_mol]
                if retro_plan["success"] is not None and retro_plan["success"]:
                    for i, (reaction, template, cost, analysis_tokens) in enumerate(
                        zip(
                            retro_plan["reaction_list"],
                            retro_plan["templates"],
                            retro_plan["cost"],
                            retro_plan["analysis_tokens"],
                        )
                    ):
                        if isinstance(analysis_tokens, torch.Tensor):
                            analysis_tokens = analysis_tokens.tolist()
                        token_list.extend(analysis_tokens + [IGNORE_INDEX])
                        text_list.extend(
                            [
                                self.tokenizer.decode(
                                    analysis_tokens,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaced=True,
                                ),
                                reaction if reaction is not None else "<NO REACTION>",
                                " with the template ",
                                template if template is not None else "<NO TEMPLATE>",
                                " which requires the reactants: ",
                            ]
                        )
                        # Add these two lines to extract and add reactants
                        if reaction is not None:
                            reactants = reaction.split(">>")[1].split(".")
                            formatted_reactants = []
                            for reactant in reactants:
                                if reactant in available_mols:
                                    formatted_reactants.append(
                                        f"{reactant} (available)"
                                    )
                                else:
                                    formatted_reactants.append(reactant)
                            text_list.extend([", ".join(formatted_reactants), ". "])
                        else:
                            text_list.extend(["<NO REACTANTS>. "])
                        ignore_positions[len(token_list) - 1] = (
                            reaction,
                            template,
                            cost,
                        )
                else:
                    analysis_tokens = retro_plan["analysis_tokens"]
                    if isinstance(analysis_tokens, torch.Tensor):
                        analysis_tokens = analysis_tokens.tolist()

                    token_list.extend(analysis_tokens)
                    text_list.extend(
                        [
                            self.tokenizer.decode(
                                analysis_tokens,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaced=True,
                            ),
                            " <NO REACTION FOUND>",
                        ]
                    )

            all_info_dict["token_lists"].append(token_list)
            all_info_dict["text_lists"].append(text_list)
            all_info_dict[f"batch_{batch_idx}_ignore_positions"] = ignore_positions

        all_info_dict["IGNORE_INDEX"] = IGNORE_INDEX
        return all_info_dict