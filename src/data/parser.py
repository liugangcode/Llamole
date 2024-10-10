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

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from ..extras.constants import DATA_CONFIG
from ..extras.misc import use_modelscope


if TYPE_CHECKING:
    from ..hparams import DataArguments


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "script", "file"]
    dataset_name: str
    formatting: Literal["alpaca", "sharegpt", "molqa"] = "molqa"
    ranking: bool = False
    # extra configs
    subset: Optional[str] = None
    folder: Optional[str] = None
    num_samples: Optional[int] = None
    # common columns
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    # rlhf columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None
    # alpaca columns
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    # sharegpt columns
    messages: Optional[str] = "conversations"
    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"
    # molqa columns
    property: Optional[str] = 'property'
    retro: Optional[str] = 'retro'
    # learned_query_size: Optional[int] = None

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))

def get_dataset_attr(data_args: "DataArguments") -> List["DatasetAttr"]:
    if data_args.dataset is not None:
        dataset_name = data_args.dataset.strip()
    else:
        raise ValueError("Please specify the dataset name.")

    try:
        with open(os.path.join(data_args.dataset_dir, DATA_CONFIG), "r") as f:
            dataset_info = json.load(f)
    except Exception as err:
        raise ValueError(
            "Cannot open {} due to {}.".format(os.path.join(data_args.dataset_dir, DATA_CONFIG), str(err))
        )
        dataset_info = None

    if dataset_name not in dataset_info:
        raise ValueError("Undefined dataset {} in {}.".format(dataset_name, DATA_CONFIG))

    dataset_attr = DatasetAttr("file", dataset_name=dataset_info[dataset_name]["file_name"])

    print('dataset_info', dataset_info)

    dataset_attr.set_attr("formatting", dataset_info[dataset_name], default="molqa")
    dataset_attr.set_attr("ranking", dataset_info[dataset_name], default=False)
    dataset_attr.set_attr("subset", dataset_info[dataset_name])
    dataset_attr.set_attr("folder", dataset_info[dataset_name])
    dataset_attr.set_attr("num_samples", dataset_info[dataset_name])

    if "columns" in dataset_info[dataset_name]:
        column_names = ["system", "tools", "images", "chosen", "rejected", "kto_tag"]
        assert dataset_attr.formatting == "molqa"
        column_names.extend(["prompt", "query", "response", "history", "property", "retro"])

        for column_name in column_names:
            dataset_attr.set_attr(column_name, dataset_info[dataset_name]["columns"])

    return dataset_attr