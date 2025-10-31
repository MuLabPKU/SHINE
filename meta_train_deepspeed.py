import hydra
import os
import torch
import tqdm
from datasets import load_dataset
from metanetwork_family_new import Metanetwork
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from typing import Any, Dict, List
from utils.mydataset import PretrainCollator, TextDataset, SquadDataset, SquadCollator
from utils.myfreeze import freeze
from LoraQwen import LoraQwen3ForCausalLM, Qwen3Config
import json
import multiprocessing as mp
from utils.mydataset import TextDataset, create_mock_dataset, SquadDataset, SquadCollator, PretrainCollator
from utils.myseed import set_seed
from utils.mylogging import get_logger
from utils.mysaveload import (
    save_checkpoint,
    load_checkpoint,
    save_training_state,
    load_training_state,
    get_latest_checkpoint,
)
from utils.myfreeze import freeze
from utils.myoptmize import init_optimize
from utils.myddp import (
    should_use_ddp,
    ddp_is_active,
    get_world_size,
    get_rank,
    get_local_rank,
    is_main_process,
    ddp_init_if_needed,
    ddp_cleanup_if_needed,
    distributed_mean,
    barrier,
)
from utils.myinit import _resolve_device, _import_class

logger = get_logger(f"metalora_debug")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers
    if cfg.metanetwork.type == "transformer":
        tmp_model = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
        lora_numel = tmp_model.lora_params_numel(cfg.model.lora_r)
        assert lora_numel % (cfg.hidden_size * cfg.num_layers) == 0, \
            "For transformer metanetwork, num_mem_token must be set to model.lora_params_numel(lora_r) / (hidden_size * num_layers)"
        config.num_mem_token = tmp_model.lora_params_numel(cfg.model.lora_r) // (cfg.hidden_size * cfg.num_layers)
        cfg.num_mem_token = config.num_mem_token
        del tmp_model
        if is_main_process():
            logger.info(f"Using transformer metanetwork, set num_mem_token to {config.num_mem_token}")
    else:
        config.num_mem_token = cfg.num_mem_token

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from)
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.train()
    freeze(metamodel) 
    
    if cfg.data.source == "transmla":
        dataset = load_dataset(os.path.join("data", "transmla_pretrain_6B_tokens"), split="train")
        split_dataset = dataset.train_test_split(test_size=0.0001, seed=42)
        train_texts = split_dataset["train"]
        val_texts = split_dataset["test"]
        train_dataset = TextDataset(train_texts["text"], tokenizer, max_length=cfg.data.max_length)
        val_dataset = TextDataset(val_texts["text"], tokenizer, max_length=cfg.data.max_length)
        collator = PretrainCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, metatrain=True)
    elif cfg.data.source == "squad":
        # features: ['id', 'title', 'context', 'question', 'answers'],
        # num_rows: 87599
        train_dataset = load_dataset(os.path.join("data", "squad"), split="train")
        val_dataset = load_dataset(os.path.join("data", "squad"), split="validation")
        train_dataset = SquadDataset(train_dataset, tokenizer, max_length=cfg.data.max_length)
        val_dataset = SquadDataset(val_dataset, tokenizer, max_length=cfg.data.max_length)
        collator = SquadCollator(tokenizer=tokenizer, max_length=cfg.data.max_length, metatrain=True)
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")
    
    training_args = TrainingArguments(
        output_dir="outputs/debug",
        overwrite_output_dir=True,
        num_train_epochs=1,
        save_strategy="no",
        per_device_train_batch_size=cfg.data.train_batch_size,
        gradient_accumulation_steps=cfg.run.gradient_accumulation_steps,
        # logging_strategy="steps",
        # logging_steps=100,
        # eval_strategy="steps",
        # eval_steps=10000,
        remove_unused_columns=False,
        weight_decay=0.01,
        deepspeed="configs/deepspeed.json",
    )
    trainer = Trainer(
        model=metanetwork,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
