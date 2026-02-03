# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reference used:
# https://medium.com/@alexmriggio/bert-for-sequence-classification-from-scratch-code-and-theory-fb88053800fa

import sys
from pathlib import Path

# Add IRON repository root to Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers import (
    BertForSequenceClassification as HFBertForSequenceClassification,
)
import logging
from safetensors.torch import load_file
import json
from types import SimpleNamespace
from datasets import load_dataset, Dataset
import random
import os
import time
from torch.utils.data import DataLoader
import argparse
from safetensors.torch import save_file
from src.model import BertForSequenceClassification
from operators.common import AIEOperatorBase

# Global logger for profiling
_profile_logger = None

SAMPLE_TEXT = """
SCENE I. King Lear's palace.
Enter KENT, GLOUCESTER, and EDMUND
KENT
I thought the king had more affected the Duke of
Albany than Cornwall.
GLOUCESTER
It did always seem so to us: but now, in the
division of the kingdom, it appears not which of
the dukes he values most; for equalities are so
weighed, that curiosity in neither can make choice
of either's moiety.
KENT
Is not this your son, my lord?
GLOUCESTER

His breeding, sir, hath been at my charge: I have
so often blushed to acknowledge him, that now I am
brazed to it.
KENT
I cannot conceive you.
GLOUCESTER
Sir, this young fellow's mother could: whereupon
she grew round-wombed, and had, indeed, sir, a son
for her cradle ere she had a husband for her bed.
Do you smell a fault?
KENT
I cannot wish the fault undone, the issue of it
being so proper.
GLOUCESTER
But I have, sir, a son by order of law, some year
elder than this, who yet is no dearer in my account:
though this knave came something saucily into the
world before he was sent for, yet was his mother
fair; there was good sport at his making, and the
whoreson must be acknowledged. Do you know this
noble gentleman, Edmund?
EDMUND
No, my lord.
GLOUCESTER
My lord of Kent: remember him hereafter as my
honourable friend.
EDMUND
My services to your lordship.
KENT
I must love you, and sue to know you better.
EDMUND
Sir, I shall study deserving.
GLOUCESTER
He hath been out nine years, and away he shall
again. The king is coming.
Sennet. Enter KING LEAR, CORNWALL, ALBANY, GONERIL, REGAN, CORDELIA, and Attendants

KING LEAR
Attend the lords of France and Burgundy, Gloucester.
GLOUCESTER
I shall, my liege.
Exeunt GLOUCESTER and EDMUND
KING LEAR
Meantime we shall express our darker purpose.
Give me the map there. Know that we have divided
In three our kingdom: and 'tis our fast intent
To shake all cares and business from our age;
Conferring them on younger strengths, while we
Unburthen'd crawl toward death. Our son of Cornwall,
And you, our no less loving son of Albany,
We have this hour a constant will to publish
Our daughters' several dowers, that future strife
May be prevented now. The princes, France and Burgundy,
Great rivals in our youngest daughter's love,
Long in our court have made their amorous sojourn,
And here are to be answer'd. Tell me, my daughters,--
Since now we will divest us both of rule,
Interest of territory, cares of state,--
Which of you shall we say doth love us most?
That we our largest bounty may extend
Where nature doth with merit challenge. Goneril,
Our eldest-born, speak first.
GONERIL
Sir, I love you more than words can wield the matter;
Dearer than eye-sight, space, and liberty;
Beyond what can be valued, rich or rare;
No less than life, with grace, health, beauty, honour;
As much as child e'er loved, or father found;
A love that makes breath poor, and speech unable;
Beyond all manner of so much I love you.
CORDELIA
[Aside] What shall Cordelia do?
Love, and be silent.
LEAR
Of all these bounds, even from this line to this,
With shadowy forests and with champains rich'd,
With plenteous rivers and wide-skirted meads,
We make thee lady: to thine and Albany's issue
Be this perpetual. What says our second daughter,
Our dearest Regan, wife to Cornwall? Speak.
REGAN
Sir, I am made
Of the self-same metal that my sister is,
And prize me at her worth. In my true heart
I find she names my very deed of love;
Only she comes too short: that I profess
Myself an enemy to all other joys,
Which the most precious square of sense possesses;
And find I am alone felicitate
In your dear highness' love.
CORDELIA
[Aside] Then poor Cordelia!
And yet not so; since, I am sure, my love's
More richer than my tongue.
KING LEAR
To thee and thine hereditary ever
Remain this ample third of our fair kingdom;
No less in space, validity, and pleasure,
Than that conferr'd on Goneril. Now, our joy,
Although the last, not least; to whose young love
The vines of France and milk of Burgundy
Strive to be interess'd; what can you say to draw
A third more opulent than your sisters? Speak.
CORDELIA
Nothing, my lord.
KING LEAR
Nothing!
CORDELIA
Nothing.
KING LEAR
Nothing will come of nothing: speak again.
CORDELIA
Unhappy that I am, I cannot heave
My heart into my mouth: I love your majesty
According to my bond; nor more nor less.
KING LEAR
How, how, Cordelia! mend your speech a little,
Lest it may mar your fortunes.
CORDELIA
Good my lord,
You have begot me, bred me, loved me: I
Return those duties back as are right fit,
Obey you, love you, and most honour you.
Why have my sisters husbands, if they say
They love you all? Haply, when I shall wed,
That lord whose hand must take my plight shall carry
Half my love with him, half my care and duty:
Sure, I shall never marry like my sisters,
To love my father all.
"""


def profile_function_calls(frame, event, arg):
    """
    Profile function that logs start and end times of every function call.

    Args:
        frame: The current stack frame
        event: The event type ('call', 'return', 'c_call', 'c_return', 'c_exception')
        arg: Event-specific argument
    """
    global _profile_logger

    if _profile_logger is None:
        return

    func_name = frame.f_code.co_name
    filename = frame.f_code.co_filename
    line_no = frame.f_lineno

    # Create a readable function identifier
    func_identifier = f"{filename}:{func_name}:{line_no}"

    if event == "call":
        # Function is being called
        timestamp = time.perf_counter()
        _profile_logger.debug(f"[CALL] {func_identifier} started at {timestamp:.9f}")

    elif event == "return":
        # Function is returning
        timestamp = time.perf_counter()
        _profile_logger.debug(f"[RETURN] {func_identifier} ended at {timestamp:.9f}")

    return profile_function_calls


def enable_profiling(logs_dir_name):
    """Enable function call profiling using sys.setprofile."""
    global _profile_logger

    # Create a dedicated logger for profiling
    _profile_logger = logging.getLogger("function_profiler")
    _profile_logger.setLevel(logging.DEBUG)
    # Prevent propagation to root logger to avoid console output
    _profile_logger.propagate = False

    # Create log file for profiling data
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        logs_dir_name,
        f"profile_{timestamp}.log",
    )

    # Add file handler for profiling (only file, no console output)
    profile_handler = logging.FileHandler(log_path)
    profile_handler.setLevel(logging.DEBUG)
    profile_formatter = logging.Formatter("%(asctime)s - %(message)s")
    profile_handler.setFormatter(profile_formatter)
    _profile_logger.addHandler(profile_handler)

    # Set the profile function
    sys.setprofile(profile_function_calls)
    _profile_logger.info("Function profiling enabled")

    # Explicitly call profile_function_calls to log this function's call
    import inspect

    frame = inspect.currentframe()
    profile_function_calls(frame, "call", None)


def disable_profiling():
    """Disable function call profiling."""
    global _profile_logger

    sys.setprofile(None)
    if _profile_logger:
        _profile_logger.info("Function profiling disabled")
        # Close all handlers
        for handler in _profile_logger.handlers[:]:
            handler.close()
            _profile_logger.removeHandler(handler)


def setup_logging(verbosity):
    """Set up logging based on verbosity level."""

    # Ensure the logs directory is created in case of profiling
    logs_dir_name = "logs"
    if not os.path.exists(logs_dir_name):
        os.makedirs(logs_dir_name)

    if verbosity != 0:
        levels = {
            4: logging.DEBUG,
            3: logging.INFO,
            2: logging.WARNING,
            # 1: log everything (DEBUG) to a file
        }

        # Create log file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/inference_{timestamp}.log"

        handlers = [logging.FileHandler(log_file)]
        if verbosity > 0:
            handlers.append(logging.StreamHandler(sys.stderr))
            handlers[-1].setLevel(levels[verbosity])

        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
            force=True,  # Override any existing configuration
        )

    return logs_dir_name


def dtype_from_string(inp):
    if isinstance(inp, torch.dtype):
        return inp
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        inp, torch.float32
    )


def load_bert_config(config_path=None):
    """Load BERT configuration from JSON file"""
    if config_path is None:
        # Default to config.json in the llama directory
        config_path = Path(__file__) / "config" / "config.json"

    with open(config_path, "r") as config_file:
        data = json.load(config_file)
        # Convert dict recursively to object with attribute-style access
        config = json.loads(
            json.dumps(data), object_hook=lambda d: SimpleNamespace(**d)
        )
        config.model_config.num_labels = data.get(
            "num_labels", 2
        )  # Default to 2 if not specified
        config.aie_config.dtype = dtype_from_string(config.aie_config.dtype)

    return config


def classify_text(model, tokenizer, text, runs_per_sample, device="cpu", seq_len=512):
    """
    Predict sentiment for input text(s).

    Parameters:
    - model: The BertForSequenceClassification model
    - tokenizer: The BERT tokenizer
    - text: Either a single string or a list of strings to classify
    - device: Device to run inference on ('cpu' or 'cuda')

    Returns:
    - probabilities: Softmax probabilities for each class
    - logits: Raw logits from the model
    """
    model.eval()
    is_single_text = isinstance(text, str)

    # Tokenize input(s)
    if is_single_text:
        # Use encode_plus for single text
        encoded = tokenizer.encode_plus(
            text,
            padding="max_length",  # Pad to max_length
            truncation=True,  # Truncate if longer than max_length
            max_length=seq_len,
            return_tensors="pt",  # Return PyTorch tensors
        )
    else:
        # Use batch_encode_plus for multiple texts
        encoded = tokenizer.batch_encode_plus(
            text,
            padding="max_length",  # Pad to max_length
            truncation=True,  # Truncate if longer than max_length
            max_length=seq_len,
            return_tensors="pt",  # Return PyTorch tensors
        )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded["token_type_ids"].to(device)

    # Print the tokens with special strings included
    # tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    # print("Tokens with special strings:", tokens)

    # Print about the tokens
    # print(input_ids[input_ids != 0].float().mean())
    # print(input_ids.shape)
    # print(attention_mask.shape)
    # print(token_type_ids.shape)
    print("attn_mask:", attention_mask.numpy())

    # # Adjust the sequence length accepted for the embeddings based on the input
    # model.bert.embeddings.position_embeddings.weight = nn.Parameter(model.bert.embeddings.position_embeddings.weight[:len(input_ids)])
    # model.bert.embeddings.position_ids = model.bert.embeddings.position_ids[:, :len(input_ids)]
    with torch.no_grad():
        avg_latency = 0
        for _ in range(runs_per_sample):
            start = time.time()
            logits = model(input_ids, token_type_ids, attention_mask=attention_mask)
            end = time.time()
            print(f"Inference time: {end - start:.4f} seconds")
            avg_latency += end - start

    # Calculate the softmax of the logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # If input was a single text, squeeze the batch dimension
    if is_single_text:
        probabilities = probabilities.squeeze(0)
        logits = logits.squeeze(0)

    return probabilities, logits, avg_latency / runs_per_sample


def fine_tune_model(
    model,
    config,
    tokenizer,
    train_dataset,
    device="cpu",
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    seq_len=512,
):
    """
    Fine-tune the BERT model on the SST-2 training dataset.

    Parameters:
    - model: The BertForSequenceClassification model
    - config: The Bert config
    - tokenizer: The BERT tokenizer
    - train_dataset: The training dataset
    - device: Device to train on ('cpu' or 'cuda')
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    - learning_rate: Learning rate for optimizer
    """
    print(f"\n{'='*50}")
    print(f"Starting Fine-Tuning on SST-2 Training Dataset")
    print(f"{'='*50}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    print(f"Training samples: {len(train_dataset)}\n")

    # Calculate mean and standard deviation of seq_relationship weights
    seq_relationship_weights = model.classifier.weight.data
    mean = seq_relationship_weights.mean().item()
    stddev = seq_relationship_weights.std().item()

    print(f"Prev Mean of seq_relationship weights: {mean:.4f}")
    print(f"Prev Standard deviation of seq_relationship weights: {stddev:.4f}")

    model.train()

    # Prepare optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Prepare data loader
    def collate_fn(batch):
        texts = [item["sentence"] for item in batch]
        labels = [item["label"] for item in batch]

        # Tokenize batch
        encoded = tokenizer.batch_encode_plus(
            texts,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "token_type_ids": encoded["token_type_ids"].to(device),
            "labels": torch.tensor(labels).to(device),
        }

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Training loop
    total_batches = len(train_loader)
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            logits = model(
                batch["input_ids"], batch["token_type_ids"], batch["attention_mask"]
            )

            # Calculate loss
            loss = criterion(logits, batch["labels"])

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

            # Print progress every 1/4 batches
            if (batch_idx + 1) % (total_batches // 4) == 0 or (
                batch_idx + 1
            ) == total_batches:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100 * correct / total
                print(
                    f"  Batch {batch_idx + 1}/{total_batches} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
                )

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{epochs} Complete - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%\n"
        )

    print(f"\n{'='*50}")
    print(f"Fine-Tuning Completed!")
    print(f"{'='*50}\n")

    # Save the fine-tuned model weights
    print("Saving fine-tuned model weights...")
    finetuned_weights = {}

    # Save embeddings
    finetuned_weights["bert.embeddings.word_embeddings.weight"] = (
        model.bert.embeddings.word_embeddings.weight.data
    )
    finetuned_weights["bert.embeddings.position_embeddings.weight"] = (
        model.bert.embeddings.position_embeddings.weight.data
    )
    finetuned_weights["bert.embeddings.token_type_embeddings.weight"] = (
        model.bert.embeddings.token_type_embeddings.weight.data
    )
    finetuned_weights["bert.embeddings.LayerNorm.beta"] = (
        model.bert.embeddings.LayerNorm.bias.data
    )
    finetuned_weights["bert.embeddings.LayerNorm.gamma"] = (
        model.bert.embeddings.LayerNorm.weight.data
    )

    # Save encoder layers
    for l in range(config.model_config.num_hidden_layers):
        finetuned_weights[f"bert.encoder.layer.{l}.attention.output.LayerNorm.beta"] = (
            model.bert.encoder.layer[l].attention.output.LayerNorm.bias.data
        )
        finetuned_weights[
            f"bert.encoder.layer.{l}.attention.output.LayerNorm.gamma"
        ] = model.bert.encoder.layer[l].attention.output.LayerNorm.weight.data
        finetuned_weights[f"bert.encoder.layer.{l}.attention.output.dense.bias"] = (
            model.bert.encoder.layer[l].attention.output.dense.bias.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.attention.output.dense.weight"] = (
            model.bert.encoder.layer[l].attention.output.dense.weight.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.attention.self.key.bias"] = (
            model.bert.encoder.layer[l].attention.self.key.bias.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.attention.self.key.weight"] = (
            model.bert.encoder.layer[l].attention.self.key.weight.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.attention.self.query.bias"] = (
            model.bert.encoder.layer[l].attention.self.query.bias.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.attention.self.query.weight"] = (
            model.bert.encoder.layer[l].attention.self.query.weight.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.attention.self.value.bias"] = (
            model.bert.encoder.layer[l].attention.self.value.bias.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.attention.self.value.weight"] = (
            model.bert.encoder.layer[l].attention.self.value.weight.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.intermediate.dense.bias"] = (
            model.bert.encoder.layer[l].intermediate.dense.bias.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.intermediate.dense.weight"] = (
            model.bert.encoder.layer[l].intermediate.dense.weight.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.output.LayerNorm.beta"] = (
            model.bert.encoder.layer[l].output.LayerNorm.bias.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.output.LayerNorm.gamma"] = (
            model.bert.encoder.layer[l].output.LayerNorm.weight.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.output.dense.bias"] = (
            model.bert.encoder.layer[l].output.dense.bias.data
        )
        finetuned_weights[f"bert.encoder.layer.{l}.output.dense.weight"] = (
            model.bert.encoder.layer[l].output.dense.weight.data
        )

    # Save pooler and classifier
    finetuned_weights["bert.pooler.dense.bias"] = model.bert.pooler.dense.bias.data
    finetuned_weights["bert.pooler.dense.weight"] = model.bert.pooler.dense.weight.data
    finetuned_weights["cls.seq_relationship.bias"] = model.classifier.bias.data
    finetuned_weights["cls.seq_relationship.weight"] = model.classifier.weight.data
    # Calculate mean and standard deviation of seq_relationship weights
    seq_relationship_weights = model.classifier.weight.data
    mean = seq_relationship_weights.mean().item()
    stddev = seq_relationship_weights.std().item()

    print(f"New Mean of seq_relationship weights: {mean:.4f}")
    print(f"New Standard deviation of seq_relationship weights: {stddev:.4f}")

    # Save to file
    save_file(finetuned_weights, "model_finetuned.safetensors")
    print("Fine-tuned model weights saved to 'model_finetuned.safetensors'")

    model.eval()
    return model


def assign(left, right, tensor_name="unknown"):
    """
    Assigns the value of the right tensor to a new torch.nn.Parameter after validating shape compatibility.

    Parameters:
    left (torch.Tensor or any): The tensor to compare shape with.
    right (torch.Tensor or any): The tensor or value to be assigned.
    tensor_name (str): The name of the tensor for error reporting (default is "unknown").

    Returns:
    torch.nn.Parameter: A new parameter containing the value of right.

    Raises:
    ValueError: If the shapes of left and right do not match.
    """

    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
        )

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="BERT inference with optional fine-tuning"
    )
    parser.add_argument(
        "weights_file_path",
        type=str,
        help="Path to the weights file: model.safetensors",
    )
    parser.add_argument(
        "config_file_path",
        type=str,
        help="Path to the config file: config.json",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to classify",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        default=False,
        help="Fine-tune the model on SST-2 training data before evaluation (default: False)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use a custom profiler for performance measurements",
    )
    parser.add_argument(
        "--runs_per_sample",
        type=int,
        default=1,
        help="Number of times to run inference with each sample (for calculating average latency)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Sequence length for BERT input",
    )
    parser.add_argument(
        "-v",
        action="count",
        default=0,
        help="Increase verbosity level (use -v (logs to file), -vv, -vvv, or -vvvv)",
    )
    args = parser.parse_args()

    # Set up logging
    logs_dir_name = setup_logging(args.v)

    # Enable function profiling
    if args.profile:
        enable_profiling(logs_dir_name)

    try:
        # Load configuration from config.json
        config = load_bert_config(args.config_file_path)
        device = "cpu"
        seq_len = args.seq_len

        # Set the max sequence length in the model config and model for compilation of operations offloaded to the AIE
        config.model_config.max_position_embeddings = seq_len
        model = BertForSequenceClassification(config, seq_len=seq_len)

        # Load the weights
        combined_weights = load_file(args.weights_file_path)
        # for key in sorted(combined_weights.keys()):
        #     val = combined_weights[key]
        #     print(key, ":", val.shape)
        # print("\n\nEmbedding weights:")
        # combined_weights_1 = load_file("model.safetensors")
        # combined_weights_2 = load_file("model_finetuned.safetensors")
        # print(combined_weights_1["bert.embeddings.word_embeddings.weight"].mean())
        # print(combined_weights_1["bert.embeddings.word_embeddings.weight"].std())
        # print(combined_weights_1["bert.embeddings.position_embeddings.weight"].mean())
        # print(combined_weights_1["bert.embeddings.position_embeddings.weight"].std())
        # print(combined_weights_1["bert.embeddings.token_type_embeddings.weight"].mean())
        # print(combined_weights_1["bert.embeddings.token_type_embeddings.weight"].std())
        # print(combined_weights_1["bert.embeddings.LayerNorm.beta"].mean())
        # print(combined_weights_1["bert.embeddings.LayerNorm.beta"].std())
        # print(combined_weights_1["bert.embeddings.LayerNorm.gamma"].mean())
        # print(combined_weights_1["bert.embeddings.LayerNorm.gamma"].std())
        # print(combined_weights_1["cls.seq_relationship.weight"].mean())
        # print(combined_weights_1["cls.seq_relationship.weight"].std())
        # print(combined_weights_1["cls.seq_relationship.bias"].mean())
        # print(combined_weights_1["cls.seq_relationship.bias"].std())
        # print(combined_weights_1["bert.pooler.dense.weight"].mean())
        # print(combined_weights_1["bert.pooler.dense.weight"].std())
        # print(combined_weights_1["bert.pooler.dense.bias"].mean())
        # print(combined_weights_1["bert.pooler.dense.bias"].std())

        # print(combined_weights_2["bert.embeddings.word_embeddings.weight"].mean())
        # print(combined_weights_2["bert.embeddings.word_embeddings.weight"].std())
        # print(combined_weights_2["bert.embeddings.position_embeddings.weight"].mean())
        # print(combined_weights_2["bert.embeddings.position_embeddings.weight"].std())
        # print(combined_weights_2["bert.embeddings.token_type_embeddings.weight"].mean())
        # print(combined_weights_2["bert.embeddings.token_type_embeddings.weight"].std())
        # print(combined_weights_2["bert.embeddings.LayerNorm.beta"].mean())
        # print(combined_weights_2["bert.embeddings.LayerNorm.beta"].std())
        # print(combined_weights_2["bert.embeddings.LayerNorm.gamma"].mean())
        # print(combined_weights_2["bert.embeddings.LayerNorm.gamma"].std())
        # print(combined_weights_2["cls.seq_relationship.weight"].mean())
        # print(combined_weights_2["cls.seq_relationship.weight"].std())
        # print(combined_weights_2["cls.seq_relationship.bias"].mean())
        # print(combined_weights_2["cls.seq_relationship.bias"].std())
        # print(combined_weights_2["bert.pooler.dense.weight"].mean())
        # print(combined_weights_2["bert.pooler.dense.weight"].std())
        # print(combined_weights_2["bert.pooler.dense.bias"].mean())
        # print(combined_weights_2["bert.pooler.dense.bias"].std())
        # print("\n\n")

        # Extend embeddings to 1024 rows
        for key in [
            "bert.embeddings.position_embeddings.weight",
        ]:
            if key in combined_weights:
                original = combined_weights[key]
                if original.shape[0] < seq_len:
                    # Repeat or pad to reach 1024 rows
                    num_repeats = (seq_len // original.shape[0]) + 1
                    extended = original.repeat(num_repeats, 1)[:seq_len]
                    combined_weights[key] = extended
                if original.shape[0] > seq_len:
                    combined_weights[key] = original[:seq_len, :]

        model.bert.embeddings.word_embeddings.weight = assign(
            model.bert.embeddings.word_embeddings.weight,
            combined_weights["bert.embeddings.word_embeddings.weight"],
            "bert.embeddings.word_embeddings.weight",
        )
        model.bert.embeddings.position_embeddings.weight = assign(
            model.bert.embeddings.position_embeddings.weight,
            combined_weights["bert.embeddings.position_embeddings.weight"],
            "bert.embeddings.position_embeddings.weight",
        )
        model.bert.embeddings.token_type_embeddings.weight = assign(
            model.bert.embeddings.token_type_embeddings.weight,
            combined_weights["bert.embeddings.token_type_embeddings.weight"],
            "bert.embeddings.token_type_embeddings.weight",
        )
        model.bert.embeddings.LayerNorm.bias = assign(
            model.bert.embeddings.LayerNorm.bias,
            combined_weights["bert.embeddings.LayerNorm.beta"],
            "bert.embeddings.LayerNorm.beta",
        )
        model.bert.embeddings.LayerNorm.weight = assign(
            model.bert.embeddings.LayerNorm.weight,
            combined_weights["bert.embeddings.LayerNorm.gamma"],
            "bert.embeddings.LayerNorm.gamma",
        )
        model.bert.encoder.assign_weights(combined_weights)
        model.bert.pooler.dense.bias = assign(
            model.bert.pooler.dense.bias,
            combined_weights["bert.pooler.dense.bias"],
            "bert.pooler.dense.bias",
        )
        model.bert.pooler.dense.weight = assign(
            model.bert.pooler.dense.weight,
            combined_weights["bert.pooler.dense.weight"],
            "bert.pooler.dense.weight",
        )
        model.classifier.bias = assign(
            model.classifier.bias,
            combined_weights["cls.seq_relationship.bias"],
            "cls.seq_relationship.bias",
        )
        model.classifier.weight = assign(
            model.classifier.weight,
            combined_weights["cls.seq_relationship.weight"],
            "cls.seq_relationship.weight",
        )
        model.to(device)
        del combined_weights

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        classes = {0: "Negative", 1: "Positive"}

        # Fine-tune the model if requested
        if args.fine_tune:
            # Load SST-2 training dataset for fine-tuning
            print("Loading SST-2 training dataset...")
            full_train_dataset = load_dataset("sst2", split="train")

            # Use only a small subset for fine-tuning (e.g., 1000 samples)
            num_train_samples = 1024
            train_dataset = full_train_dataset.select(
                range(min(num_train_samples, len(full_train_dataset)))
            )
            print(
                f"Using {len(train_dataset)} samples for fine-tuning (subset of {len(full_train_dataset)} total)"
            )

            # Fine-tune the model on SST-2 training data
            model = fine_tune_model(
                model=model,
                config=config,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                device=device,
                epochs=5,
                batch_size=128,
                learning_rate=2e-5,
                seq_len=seq_len,
            )
        else:
            print("Skipping fine-tuning (use --fine-tune flag to enable)")

        # Important: Set the seed again after initialization of the model. Each
        # call that initializes an nn.Linear layer updates the RNG state, because
        # weights are initialized with random values. For different JSON
        # configurations, we initialize a different number of linear layers,
        # so different configurations result in a different RNG state here. Since
        # we use random numbers to sample from the token distribution during
        # inference, it is important to have the same RNG state between runs so we
        # can have reproducible results across configurations.
        torch.manual_seed(1608560892)

        # At this point the model is fully described (operators and their dimensions and how to compile them)
        AIEOperatorBase.get_default_context().compile_all()
        AIEOperatorBase.get_default_context().prepare_runtime()
        logging.info("AIE operator preparation completed.")

        # Load validation dataset for evaluation
        print("Loading SST-2 validation dataset for evaluation...")
        num_samples_to_test = args.num_samples
        if num_samples_to_test == 1:
            # NOTE: Using text that generates at least 512 tokens so that there's no attention masking
            texts_to_classify = [SAMPLE_TEXT]
            true_labels = [0]  # Negative
        else:
            dataset = load_dataset("sst2", split="validation")

            # Half of dataset should be positive and other half negative
            sorted_dataset = dataset.sort("label")
            texts_to_classify = dataset["sentence"][: num_samples_to_test // 2]
            texts_to_classify = (
                texts_to_classify + dataset["sentence"][-num_samples_to_test // 2 :]
            )
            true_labels = dataset["label"][: num_samples_to_test // 2]
            print(true_labels[:5])
            true_labels = true_labels + dataset["label"][-num_samples_to_test // 2 :]
            print(true_labels[num_samples_to_test // 2 : num_samples_to_test // 2 + 5])

        correct_predictions = []
        wrong_predictions = []

        iteration_count = 0
        # Process each text individually
        total_inference_time = 0.0
        for test_text, true_class in zip(texts_to_classify, true_labels):
            iteration_count = iteration_count + 1
            probabilities, logits, inference_time = classify_text(
                model, tokenizer, test_text, args.runs_per_sample, device, seq_len
            )
            total_inference_time += inference_time
            predicted_class = probabilities.argmax().item()
            print(probabilities, logits)
            print(
                f"Iteration {iteration_count}: Text: Predicted Class: {classes[predicted_class]} | True Class: {classes[true_class]}"
            )

            if predicted_class == true_class:
                if len(correct_predictions) < 5:
                    correct_predictions.append((test_text, predicted_class))
            else:
                if len(wrong_predictions) < 5:
                    wrong_predictions.append((test_text, predicted_class, true_class))

            # print("Output:", probabilities, logits)
            # print(test_text, "Classified as", classes[predicted_class])

        print("\nCorrect Predictions:")
        for text, pred in correct_predictions:
            print(f"Text: {text} | Predicted Class: {classes[pred]}")

        print("\nWrong Predictions:")
        for text, pred, true in wrong_predictions:
            print(
                f"Text: {text} | Predicted Class: {classes[pred]} | True Class: {classes[true]}"
            )

        total_predictions = len(correct_predictions) + len(wrong_predictions)
        accuracy = (
            len(correct_predictions) / total_predictions if total_predictions > 0 else 0
        )
        print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

        average_inference_time = total_inference_time / num_samples_to_test * 1000
        print(
            f"Average Inference Time per Sample: {average_inference_time:.4f} milliseconds"
        )

        # Clean the dataset cache after the run
        # dataset.cleanup_cache_files()
    finally:
        if args.profile:
            # Disable profiling when done
            disable_profiling()


if __name__ == "__main__":
    main()
