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

Tools for streaming and working with large datasets.
"""

import logging
import os

from datasets import load_dataset

logger = logging.getLogger(__name__)


def prepare_dataset(tokenizer, config, max_seq_len, tokenization_args):
    """Prepare dataset from list of files.

    Args:
        tokenizer: Tokenizer to apply on the files.
        config: Configuration json for the data files.
        max_seq_len: Maximum sequence length.
        tokenization_args: Addtional arguments to be passed to tokenizer's call function. Truncation and max_length are set by default.

    Returns:
        dataset: HuggingFace dataset object.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset = load_dataset(config["dataset_name"], split="train").shuffle(seed=41).select(range(51_200_000))
    column = config.get("column", "text")

    def tokenize_fn(examples):
        text = ["\n".join(e) for e in examples[column]]
        return tokenizer(
            text, truncation=True, max_length=max_seq_len, **tokenization_args
        )

    return dataset.map(tokenize_fn, batched=True).with_format("torch")


def get_tokenized_datasets(data_args, tokenizer, tokenization_args=None):
    """Get the tokenized train and dev datasets.

    Args:
        data_args: Arguments from data parser.
        tokenizer: Tokenizer to apply on the datasets.
        tokenization_args: Addtional arguments to be passed to tokenizer's call function. Truncation and max_length are set by default.
    Returns:`
        Tuple of train and val tokenized datasets.
    """
    if not tokenization_args:
        tokenization_args = {}
    train_tokenized_dataset = prepare_dataset(
        tokenizer,
        data_args.train_config,
        data_args.max_seq_len,
        tokenization_args,
    )
    val_tokenized_dataset = (
        prepare_dataset(
            tokenizer,
            data_args.val_config,
            data_args.max_seq_len,
            tokenization_args,
        )
        if data_args.val_config
        else None
    )

    return train_tokenized_dataset, val_tokenized_dataset
