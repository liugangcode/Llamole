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

from typing import TYPE_CHECKING, List, Optional

from ...data import get_dataset, split_dataset, DataCollatorForSeqGraph
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_language_model, load_tokenizer

from ...model import GraphLLMForCausalMLM

from .metric import ComputeMetrics, compute_accuracy, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
    )


def run_mmsft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    mol_id_to_pyg, dataset = get_dataset(
        model_args, data_args, training_args, tokenizer=tokenizer
    )
    
    data_collator = DataCollatorForSeqGraph(
        tokenizer=tokenizer,
        mol_id_to_pyg=mol_id_to_pyg,
        pad_to_multiple_of=(
            8 if tokenizer.padding_side == "right" else None
        ),  # for shift short attention
        label_pad_token_id=(
            IGNORE_INDEX
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id
        ),
    )

    model = GraphLLMForCausalMLM.from_pretrained(
        tokenizer, model_args, data_args, training_args, finetuning_args
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length or data_args.cutoff_len
    )
    training_args.generation_num_beams = (
        data_args.eval_num_beams or training_args.generation_num_beams
    )
    training_args.remove_unused_columns = False
    
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=(
            ComputeMetrics(tokenizer)
            if training_args.predict_with_generate
            else compute_accuracy
        ),
        preprocess_logits_for_metrics=(
            None if training_args.predict_with_generate else eval_logit_processor
        ),
        **tokenizer_module,
        **split_dataset(dataset, data_args, training_args),
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [
        tokenizer.eos_token_id
    ] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(
                training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"]
            )