"""Copyright 2022 Bloomberg Finance L.P.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Contains code to perform distillation over a large corpus (Wikipedia + bookcorpus) using MiniLMv2.
"""

import json
import logging
import os
import sys
from ast import literal_eval

import pkg_resources
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from .minilmv2 import MiniLM
from .parsers import get_data_parser, get_model_parser, split_args_by_parser

logger = logging.getLogger(__name__)


def _get_args():
    data_parser = get_data_parser()
    model_parser = get_model_parser()
    training_parser = HfArgumentParser((TrainingArguments))

    parsers = {
        "data_params": data_parser,
        "model_params": model_parser,
        "training_params": training_parser,
    }
    params = split_args_by_parser(sys.argv[1:], parsers)
    params["training_params"].label_names = ["start_positions", "end_positions"]
    params["training_params"].local_rank = _get_rank() if _is_distributed() else -1
    hf_training_args = TrainingArguments(**vars(params["training_params"]))
    data_args = params["data_params"]
    model_args = params["model_params"]
    return data_args, model_args, hf_training_args


def _is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _get_rank():
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def main():
    """Main entry point for running distillation."""
    data_args, model_args, hf_training_args = _get_args()

    data_args.train_config = json.loads(
        pkg_resources.resource_string(__name__, data_args.train_config)
    )
    data_args.val_config = (
        json.loads(pkg_resources.resource_string(__name__, data_args.val_config))
        if data_args.val_config
        else None
    )

    # Set seed before initializing model.
    set_seed(hf_training_args.seed)

    input_model_dir = model_args.input_model_dir

    checkpoint_dir = model_args.checkpoint_dir
    tokenizer_dir = (
        model_args.tokenizer_dir if model_args.tokenizer_dir else input_model_dir
    )

    # Teacher
    logger.info("Loading Teacher from pretrained model")
    teacher = AutoModel.from_pretrained(input_model_dir)
    logger.info("Loaded Teacher")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, use_fast=True, max_length=data_args.max_seq_len
    )

    # Student
    logger.info("Initializing student model")
    student_config = AutoConfig.from_pretrained(input_model_dir)
    student_config.hidden_size = model_args.student_hidden_size
    student_config.num_hidden_layers = model_args.student_num_layers
    student_config.num_attention_heads = model_args.student_attention_heads

    logger.info("Student Configuration")
    logger.info(student_config)
    student = AutoModel.from_config(student_config)

    logger.info("Initializing MiniLMv2")
    # Note: change the hyperparameter minilm_relations in line if you need
    # The format is {(relation id1, relation id2): weight}
    # Relation ids are denoted as 1: Query, 2: Key, 3: Value
    minilm_relations = literal_eval(model_args.minilm_relations)
    logger.info(f"MiniLM relations: {minilm_relations}")
    distiller = MiniLM(
        teacher=teacher,
        student=student,
        L=model_args.L,
        M=model_args.student_num_layers,
        relations=minilm_relations,
        A_r=model_args.num_relation_heads,
    )

    if checkpoint_dir is not None:
        logger.info("Loading model from checkpoint")
        distiller_state_dict = torch.load(checkpoint_dir + "/pytorch_model.bin")
        distiller.load_state_dict(distiller_state_dict)
        logger.info("Loaded checkpoint")

        student_output_dir = checkpoint_dir + "/student"
        os.makedirs(student_output_dir, exist_ok=True)

        torch.save(distiller.student.state_dict(), student_output_dir + "/pytorch_model.bin")
        student_config.save_pretrained(student_output_dir)
        tokenizer.save_pretrained(student_output_dir)

        print("---- DONE -----")


if __name__ == "__main__":
    main()
