#!/usr/bin/env python
# -*- coding: utf-8 -*-

from csv import writer
import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from functools import partial
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset as HFDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import importlib
from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset
import logging
from torch.utils.tensorboard import SummaryWriter
from metanetwork_family import Metanetwork

from utils.mydataset import TextDataset, create_mock_dataset, SquadDataset, SquadCollator, PretrainCollator, GroupedSquadDataset, SFTDataset, SFTCollator
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
from collections import OrderedDict
from typing import Optional, Union, Mapping, Sequence

logger = get_logger("metalora")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# @torch.no_grad()
# def generate_stepwise(
#     model,
#     tokenizer,
#     input_ids: torch.LongTensor,                     # [1, T]
#     labels: Optional[torch.LongTensor] = None,  # [1, T]
#     attention_mask: Optional[torch.LongTensor] = None,
#     max_new_tokens: int = 128,
#     eos_token_id: Optional[Union[int, List[int]]] = None,
#     do_sample: bool = False,
#     temperature: float = 1.0,
#     top_p: float = 1.0,
#     top_k: int = 0,
#     repetition_penalty: float = 1.0,
#     # metanet extras (ignored by plain HF models)
#     loradict=None,
#     ignore_mem_token: Optional[bool] = True,
#     # amp
#     use_amp: bool = False,
#     device: Optional[torch.device] = None,
# ):
#     """
#     Yields a dict per decoding step:
#       {
#         'step': int,
#         'chosen_id': int,
#         'chosen_token': str,
#         'chosen_prob': float,
#         'top5': List[{'id': int, 'token': str, 'prob': float}],
#         'all_ids': torch.LongTensor,  # current full sequence [T_prompt + t]
#       }
#     The 'top5' list is computed from the *effective* distribution used to choose the token.
#     """
#     model_was_training = model.training
#     model.eval()

#     if device is None:
#         device = input_ids.device
#     if attention_mask is None:
#         attention_mask = torch.ones_like(input_ids, device=device)
#     if eos_token_id is None:
#         eos_token_id = tokenizer.eos_token_id
#     eos_set = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])

#     # Flip cache on for fast incremental decoding
#     _old_cache = getattr(getattr(model, "config", object()), "use_cache", None)
#     if hasattr(model, "config"):
#         try:
#             model.config.use_cache = True
#         except Exception:
#             pass

#     generated = input_ids.clone()   # [1, T]
#     if generated.dim() != 2 or generated.size(0) != 1:
#         raise ValueError("Please pass input_ids of shape [1, T].")

#     if attention_mask.dim() != 2 or attention_mask.size(0) != 1:
#         raise ValueError("Please pass attention_mask of shape [1, T].")

#     past_key_values = None
#     amp_ctx = (
#         torch.amp.autocast(device_type="cuda")
#         if (use_amp and device.type in ("cuda", "mps"))
#         else torch.amp.autocast(enabled=False, device_type="cpu")
#     )

#     def apply_repetition_penalty_(
#         logits: torch.Tensor, seen_ids: List[int], penalty: float
#     ):
#         if penalty == 1.0 or len(seen_ids) == 0:
#             return
#         # Simple version: divide logits of seen tokens by penalty
#         # (works well enough; more sophisticated versions treat positive vs negative logits differently)
#         unique = torch.unique(torch.tensor(seen_ids, device=logits.device))
#         logits[unique] = logits[unique] / penalty

#     def effective_probs(
#         last_logits: torch.Tensor,    # [V]
#         seen_ids: List[int],
#         temperature: float,
#         top_k: int,
#         top_p: float,
#         repetition_penalty: float,
#     ) -> torch.Tensor:
#         """Return renormalized probabilities after temp/rep-penalty/top-k/top-p."""
#         logits = last_logits.clone()

#         # repetition penalty
#         apply_repetition_penalty_(logits, seen_ids, repetition_penalty)

#         # temperature
#         if temperature and temperature > 0.0:
#             logits = logits / temperature
#         else:
#             # treat 0/None as greedy: very low temp approximates argmax
#             # still produce a valid prob distribution
#             pass

#         # Top-k: keep only k largest logits (before softmax)
#         if top_k and top_k > 0 and top_k < logits.numel():
#             kth_vals, kth_idx = torch.topk(logits, k=top_k)
#             mask = torch.full_like(logits, float("-inf"))
#             mask[kth_idx] = kth_vals
#             logits = mask

#         # Softmax first to get probs (we'll nucleus-mask on probs)
#         probs = torch.softmax(logits, dim=-1)

#         # Top-p (nucleus): keep smallest set with cumulative prob >= p
#         if 0.0 < top_p < 1.0:
#             sorted_probs, sorted_idx = torch.sort(probs, descending=True)
#             cumsum = torch.cumsum(sorted_probs, dim=-1)
#             # keep up to and including the first index where cumsum > top_p
#             cut_idx = torch.searchsorted(cumsum, torch.tensor(top_p, device=probs.device), right=True)
#             keep = sorted_idx[:cut_idx + 1]
#             # zero everything else, then renormalize
#             masked = torch.zeros_like(probs)
#             masked[keep] = probs[keep]
#             s = masked.sum()
#             if s.item() > 0:
#                 probs = masked / s

#         return probs

#     # Prime the cache with the full prompt
#     with amp_ctx:
#         kwargs = dict(input_ids=generated, attention_mask=attention_mask)
#         kwargs["labels"] = labels
#         if loradict is not None:
#             kwargs["loradict"] = loradict
#         if ignore_mem_token is not None:
#             kwargs["ignore_mem_token"] = ignore_mem_token
#         out = model(**kwargs)
#         logits = out.logits                    # [1, T, V]
#         past_key_values = getattr(out, "past_key_values", None)

#     # step-by-step decode
#     for step in range(1, max_new_tokens + 1):
#         last_logits = logits[:, -1, :].squeeze(0)   # [V]
#         seen = generated[0].tolist()

#         # Build effective distribution AFTER all knobs
#         probs = effective_probs(
#             last_logits, seen_ids=seen, temperature=temperature,
#             top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty
#         )

#         # Top-5 from the effective distribution
#         k = min(5, probs.numel())
#         top5_probs, top5_ids = torch.topk(probs, k=k, dim=-1)
#         top5_list = []
#         for pid, p in zip(top5_ids.tolist(), top5_probs.tolist()):
#             tok = tokenizer.decode([pid], skip_special_tokens=False)
#             top5_list.append({"id": pid, "token": tok, "prob": float(p)})

#         # Choose next token
#         if do_sample:
#             # sample from effective distribution
#             next_token_id = int(torch.multinomial(probs, num_samples=1).item())
#             chosen_prob = float(probs[next_token_id].item())
#         else:
#             # greedy from effective distribution (equivalent to argmax of adjusted logits)
#             next_token_id = int(top5_ids[0].item())
#             chosen_prob = float(top5_probs[0].item())

#         next_token = torch.tensor([[next_token_id]], device=device, dtype=generated.dtype)
#         generated = torch.cat([generated, next_token], dim=1)
#         attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

#         res_dict = {
#             "step": step, 
#             "chosen_id": next_token_id,
#             "chosen_token": tokenizer.decode([next_token_id], skip_special_tokens=False),
#             "chosen_prob": chosen_prob,
#             "top5": top5_list,
#         }
#         # for i in res_dict.items():
#         #     print(f"{i[0]}: {i[1]}")

#         if next_token_id in eos_set:
#             break

#         # next incremental forward with pkv
#         with amp_ctx:
#             kwargs = dict(
#                 input_ids=next_token,
#                 attention_mask=attention_mask,
#                 past_key_values=past_key_values,
#                 labels=labels,
#             )
#             if loradict is not None:
#                 kwargs["loradict"] = loradict
#             if ignore_mem_token is not None:
#                 kwargs["ignore_mem_token"] = ignore_mem_token
#             out = model(**kwargs)
#             logits = out.logits
#             past_key_values = getattr(out, "past_key_values", past_key_values)

#     # restore flags
#     if _old_cache is not None and hasattr(model, "config"):
#         try:
#             model.config.use_cache = _old_cache
#         except Exception:
#             pass
#     if model_was_training:
#         model.train()
    
#     return generated

@torch.no_grad()
def evaluate(metanetwork_ddp_or_module, dataloader, device, use_amp: bool = False, use_metanet: bool = True, metalora: Optional[torch.Tensor] = None, amp_dtype=None) -> Dict[str, float]:
    # Handle both wrapped and unwrapped metanetwork
    metanet = metanetwork_ddp_or_module.module if isinstance(metanetwork_ddp_or_module, DDP) else metanetwork_ddp_or_module
    metanet.eval()

    if use_metanet:
        assert metalora is not None, "metalora cannot be None when use_metanet is True"

    total_loss = 0.0
    total_reg_loss = 0.0
    n_tokens = 0
    amp_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype=(amp_dtype or torch.bfloat16),
            enabled=(use_amp and device.type == "cuda")
        )
    
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
        
        with amp_ctx:
            outputs = metanet(
                input_ids=input_ids,
                input_attention_mask=input_attention_mask,
                evidence_ids=evidence_ids,
                evidence_attention_mask=evidence_attention_mask,
                labels=labels,
                use_metanet=use_metanet,
                metalora=metalora,
            )
        loss = outputs.loss
        reg_loss = outputs.reg_loss

        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        if use_metanet:
            total_reg_loss += reg_loss.item() * valid_tokens
        n_tokens += valid_tokens

    # Reduce across ranks
    if ddp_is_active():
        t = torch.tensor([total_loss, n_tokens, total_reg_loss], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss = float(t[0].item())
        n_tokens = int(t[1].item())
        if use_metanet:
            total_reg_loss = float(t[2].item())

    avg_loss = total_loss / max(n_tokens, 1)
    avg_reg_loss = total_reg_loss / max(n_tokens, 1) if use_metanet else None
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")


    # batch = next(iter(test_loader))
    # evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
    # evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
    # input_ids = batch["input_ids"].to(device, non_blocking=True)
    # input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)
    # ground_truths = batch["answers"]
    # questions = batch["questions"]
    # loradict = metanet.generate_lora_dict(input_ids=evidence_ids, attention_mask=evidence_attention_mask)
    # gen_out = metanet.metamodel.generate(
    #     input_ids=input_ids,
    #     attention_mask=input_attention_mask,
    #     loradict=loradict,
    #     ignore_mem_token=True,
    #     max_new_tokens=1000,
    #     do_sample=False,
    #     # **gen_kwargs,
        
    #     # return_dict_in_generate=True,
    #     # output_scores=True
    # )
    # full_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    # if is_main_process():
    #     print(full_text)
    #     print("ground truth: " + ground_truths[0])
    #     print("######################################################")
    
    # if is_main_process():
    #     batch = next(iter(dataloader))
    #     evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)[0]
    #     evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)[0]
    #     input_ids = batch["input_ids"].to(device, non_blocking=True)[0]
    #     input = tokenizer.decode(input_ids)
    #     input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)[0]
    #     thinkend_token_id = tokenizer.convert_tokens_to_ids("</think>")
    #     for j in range(len(input_ids) - 1, -1, -1):
    #         if input_ids[j].item() == thinkend_token_id:
    #             input_ids = input_ids[:j+2]
    #             input_attention_mask = input_attention_mask[:j+2]
    #             break
    #     ground_truth = batch["answers"][0]
    #     question = batch["questions"][0]
    #     if use_metanet:
    #         loradict = metanet.generate_lora_dict(evidence_ids=evidence_ids.unsqueeze(0), evidence_attention_mask=evidence_attention_mask.unsqueeze(0))
    #     else:
    #         loradict = None
    #     print("generate_stepwise##############################################")
    #     gen_out = generate_stepwise(
    #         metanet.metamodel,
    #         tokenizer,
    #         input_ids=input_ids.unsqueeze(0),
    #         attention_mask=input_attention_mask.unsqueeze(0),
    #         loradict=loradict,
    #         ignore_mem_token=True,
    #         max_new_tokens=1000,
    #         do_sample=False,
    #         use_amp=use_amp,
    #     )
    #     full_text = tokenizer.decode(gen_out[0], skip_special_tokens=False)
    #     print("######################################################")
    #     print(input)
    #     print("######################################################")
    #     print(full_text)
    #     print("ground truth: " + ground_truth)
    #     print("######################################################")
    # dist.barrier()

    metanet.train()
    return {"eval_loss": avg_loss, "perplexity": ppl, "eval_reg_loss": avg_reg_loss}


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    amp_dtype = torch.bfloat16
    
    torch.set_float32_matmul_precision('high')
    if cfg.run.use_gradient_checkpoint: 
        torch._dynamo.config.optimize_ddp = False
    if cfg.mode == "train":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # ========= DDP init (safe for single-process) =========
    ddp_init_if_needed()
    
    if is_main_process():
        logger.info("Resolved config:")
        logger.info(f"\n\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Seed & device
    # Make seed rank-dependent to vary shuffles but keep reproducibility per rank
    set_seed(int(cfg.run.seed) + get_rank())
    device = _resolve_device(cfg.run.device)
    torch.backends.cudnn.benchmark = True
    
    # Load model/tokenizer (supports your local LoRA-wrapped Qwen class)
    if is_main_process():
        logger.info("Loading model & tokenizer...")
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from, padding_side="left")
    tokenizer.add_tokens(['<RECON>', '<COMP>'])
    tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if (loop.last or (not loop.last and reasoning_content)) and (enable_thinking is not defined or enable_thinking != false) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is not defined or enable_thinking != false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.train()
    metanetwork.to(device)
    freeze(metamodel) 
    
    # Training loop scaffolding
    ckpt_root = os.path.join("checkpoints", f"{cfg.name}", f"{cfg.mode}")
    if is_main_process():
        os.makedirs(ckpt_root, exist_ok=True)
    if cfg.resume_global_step == -1:
        resume_dir = None
    elif cfg.resume_global_step == "latest":
        resume_dir = get_latest_checkpoint(ckpt_root)
    elif isinstance(cfg.resume_global_step, int) and cfg.resume_global_step > 0:
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.resume_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    else:
        raise ValueError(f"Invalid resume_global_step: {cfg.resume_global_step}")
    
    resume_state = None
    if resume_dir is not None:
        # Load model & tokenizer
        if is_main_process():
            logger.info(f"Resume mode, loading from {resume_dir}...")
        metanetwork, metalora = load_checkpoint(metanetwork, resume_dir, device)
        resume_state = load_training_state(resume_dir)
    else:
        if cfg.mode == "train":
            try:
                pretrain_dir = os.path.join("checkpoints", f"{cfg.name}", "pretrain")
                pretrain_dir = get_latest_checkpoint(pretrain_dir)
                metanetwork, metalora = load_checkpoint(metanetwork, pretrain_dir, device)
                if is_main_process():
                    logger.info(f"Loaded metanetwork from pretrain checkpoint. {pretrain_dir}")
            except Exception as e:
                if is_main_process():
                    logger.info("No pretrain checkpoint found, initializing metanetwork from scratch.")
                metalora = metanetwork.metamodel.init_lora_dict(cfg.model.metalora_r, scale=cfg.metanetwork.transformer_cfg.scale, device=device)
        else:
            # Initialize metalora
            metalora = metanetwork.metamodel.init_lora_dict(cfg.model.metalora_r, scale=cfg.metanetwork.transformer_cfg.scale, device=device)
        
    metanetwork.metamodel.config.use_cache = False

    # ====== Wrap ONLY the trainable module in DDP when applicable ======
    metanetwork.to(device)
    if should_use_ddp():
        ddp_metanet = DDP(
            metanetwork,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    else:
        ddp_metanet = metanetwork  # no wrapping in single-process run

    # Optimizer & Scheduler
    def iter_learnable_tensors(tree, prefix="root"):
        """Yield leaf te nsors with requires_grad=True from nested dict/list/tuple, 
        and print non-leaf tensor info."""
        if isinstance(tree, Mapping):
            for k, v in tree.items():
                yield from iter_learnable_tensors(v, prefix=f"{prefix}.{k}")
        elif isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
            for i, v in enumerate(tree):
                yield from iter_learnable_tensors(v, prefix=f"{prefix}[{i}]")
        elif torch.is_tensor(tree):
            if tree.requires_grad:
                if tree.is_leaf:
                    yield tree
                else:
                    print(f"⚠️ Non-leaf tensor at path '{prefix}': "
                        f"shape={tuple(tree.shape)}, "
                        f"grad_fn={tree.grad_fn}")
                    # optionally still yield it or raise
                    raise ValueError(f"Found non-leaf tensor at '{prefix}'")
    # else: ignore other types
    if is_main_process():
        logger.info("Setting up optimizer & scheduler...")
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight", "norm1", "norm2"]
    grouped_params = [
        {
            "params": [p for n, p in ddp_metanet.named_parameters() if (not any(nd in n for nd in no_decay) and not n.startswith("module.metamodel"))],
            "weight_decay": cfg.optim.weight_decay,
        },
        {
            "params": [p for n, p in ddp_metanet.named_parameters() if (any(nd in n for nd in no_decay) and not n.startswith("module.metamodel"))],
            "weight_decay": 0.0,
        },
        {
            "params": iter_learnable_tensors(metalora),
            "weight_decay": cfg.optim.weight_decay,
        }
        # mem_tokens are already part of metanetwork's parameters
    ]  
    
    # Data
    if is_main_process():
        logger.info("Preparing data...")
    if cfg.data.source == "transmla":
        dataset = load_dataset(os.path.join("data", "transmla_pretrain_6B_tokens"), split="train")
        split_dataset = dataset.train_test_split(test_size=0.0001, seed=42)
        train_texts = split_dataset["train"]
        val_texts = split_dataset["test"]
        if is_main_process():
            logger.info(f"Train len: {len(train_texts)}")
            logger.info(f"Val len: {len(val_texts)}")
        train_ds = TextDataset(train_texts["text"], tokenizer)
        val_ds = TextDataset(val_texts["text"], tokenizer)
        train_collator = PretrainCollator(tokenizer=tokenizer, metatrain=True, cfg=cfg, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length)
        val_collator = PretrainCollator(tokenizer=tokenizer, metatrain=True, cfg=cfg, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length)
    elif cfg.data.source == "squad":
        # features: ['id', 'title', 'context', 'question', 'answers'],
        # num_rows: 87599
        train_dataset = load_dataset(os.path.join("data", "squad"), split="train")
        val_dataset = load_dataset(os.path.join("data", "squad"), split="validation")
        val_dataset = val_dataset.shuffle(seed=42).select(range(1000))
        # train_ds = SquadDataset(train_dataset, tokenizer)
        # val_ds = SquadDataset(val_dataset, tokenizer)
        train_ds = GroupedSquadDataset(train_dataset, tokenizer, 512, name="Train")
        val_ds = GroupedSquadDataset(val_dataset, tokenizer, 512, name="Validation")
        train_collator = SquadCollator(tokenizer=tokenizer, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length, metatrain=True)
        val_collator = SquadCollator(tokenizer=tokenizer, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length, metatrain=True)
    elif cfg.data.source == "sft-ift":
        dataset = load_dataset("json", data_files=os.path.join("data", "sft-ift.jsonl"))
        train_ds = SFTDataset(dataset['train'], tokenizer)
        val_dataset = load_dataset(os.path.join("data", "squad"), split="validation")
        val_dataset = val_dataset.shuffle(seed=42).select(range(1000))
        val_ds = GroupedSquadDataset(val_dataset, tokenizer, 512, name="Validation")
        train_collator = SFTCollator(tokenizer=tokenizer, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length)
        val_collator = SquadCollator(tokenizer=tokenizer, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length, metatrain=True)
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")

    

    pin = (device.type == "cuda")

    # Distributed samplers (only if world_size > 1)
    train_sampler = DistributedSampler(train_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, seed=cfg.run.seed) if get_world_size() > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=False) if get_world_size() > 1 else None

    # Use a few workers by default when on GPU
    num_workers_default = 2 if device.type == "cuda" else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.train_batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=train_collator,
        pin_memory=pin,
        num_workers=getattr(cfg.data, "num_workers", num_workers_default),
        persistent_workers=pin and getattr(cfg.data, "num_workers", num_workers_default) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=val_collator,
        pin_memory=pin,
        num_workers=getattr(cfg.data, "num_workers", num_workers_default),
        persistent_workers=pin and getattr(cfg.data, "num_workers", num_workers_default) > 0,
    )

    
    optimizer, lr_scheduler = init_optimize(grouped_params, train_loader, cfg, device)

    # Only main process writes TB logs
    tb_log_dir = os.path.join("tensorboard", f"{cfg.name}", f"{cfg.mode}")
    writer = SummaryWriter(log_dir=tb_log_dir) if is_main_process() else None
    if is_main_process():
        logger.info(f"TensorBoard logs will be written to: {tb_log_dir}")
        logger.info("Starting training loop...")


    # Make sure all ranks see the directory
    if ddp_is_active():
        dist.barrier()

    global_step = 0
    best_eval_loss = float("inf")
    start_epoch = 0
    start_step_in_epoch = 0
    if resume_state is not None:
        global_step = resume_state["global_step"]
        best_eval_loss = resume_state["best_eval_loss"]
        start_epoch = resume_state["epoch"]
        start_step_in_epoch = resume_state["step_in_epoch"]

    def one_train_epoch(epoch, start_epoch=1, start_step_in_epoch=0):
        nonlocal global_step, best_eval_loss
        epoch_loss = 0.0
        epoch_tokens = 0
        tmp_loss = 0.0
        tmp_tokens = 0
        # need to change
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        
        if epoch < start_epoch:
            for step, batch in enumerate(train_loader, start=1):
                if step % max(1, cfg.run.gradient_accumulation_steps) == 0:
                    lr_scheduler.step()
            return 

        pbar = train_loader
        if is_main_process():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.num_epochs}")

        for step, batch in enumerate(pbar, start=1):
            if epoch == start_epoch and step <= start_step_in_epoch:
                if step % max(1, cfg.run.gradient_accumulation_steps) == 0:
                    lr_scheduler.step()
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
            evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast(enabled=(cfg.run.use_amp and device.type == "cuda"), device_type="cuda", dtype=amp_dtype):
                # Forward through possibly DDP-wrapped metanetwork
                outputs = ddp_metanet(input_ids=input_ids, input_attention_mask=input_attention_mask, 
                                    evidence_ids=evidence_ids, evidence_attention_mask=evidence_attention_mask, 
                                    labels=labels, metalora=metalora, use_gradient_checkpoint=cfg.run.use_gradient_checkpoint)
                loss = (outputs.loss / max(1, cfg.run.gradient_accumulation_steps)).item()
                reg_loss = (outputs.reg_loss / max(1, cfg.run.gradient_accumulation_steps)).item()
                backward_loss = (outputs.loss + outputs.reg_loss) / max(1, cfg.run.gradient_accumulation_steps)

            if writer is not None:
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)

            backward_loss.backward()

            valid_tokens = (labels != -100).sum().item()
            # Track per-rank; we’ll reduce for logging only
            epoch_loss += loss * valid_tokens * max(1, cfg.run.gradient_accumulation_steps)
            tmp_loss += loss * valid_tokens * max(1, cfg.run.gradient_accumulation_steps)
            epoch_tokens += valid_tokens
            tmp_tokens += valid_tokens
            if math.isinf(tmp_loss) or math.isnan(tmp_loss):
                logger.info("NaN/Inf loss detected.")
                logger.info(f"Batch:\n{batch}")
                logger.info(f"loss: {loss}")
                logger.info(f"Tmp loss: {tmp_loss}")
                logger.info(f"Epoch loss: {epoch_loss}")
                logger.info(f"Valid tokens: {valid_tokens}")
                logger.info(f"Tmp tokens: {tmp_tokens}")
                logger.info(f"Epoch tokens: {epoch_tokens}")

            if step % max(1, cfg.run.gradient_accumulation_steps) == 0 or step == len(train_loader):
                if cfg.optim.grad_clip_norm and cfg.optim.grad_clip_norm > 0:
                    for group in optimizer.param_groups:
                        # # ---- Compute and print grad norm BEFORE clipping ----
                        # total_norm = 0.0
                        # for p in group["params"]:
                        #     if p.grad is not None:
                        #         param_norm = p.grad.data.norm(2)
                        #         total_norm += param_norm.item() ** 2
                        # total_norm = total_norm ** 0.5
                        # print(f"Gradient norm before clipping: {total_norm:.4f}")
                        # # ------------------------------------------------------
                        torch.nn.utils.clip_grad_norm_(group["params"], cfg.optim.grad_clip_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1

                # Periodic logging (only on rank 0, with distributed averages)
                if cfg.logging.logging_steps and global_step % cfg.logging.logging_steps == 0:
                    # everyone computes + participates in the reduction
                    avg_loss_local = (epoch_loss / max(epoch_tokens, 1))
                    tmp_loss_local = (tmp_loss / max(tmp_tokens, 1))
                    avg_loss_world = distributed_mean(avg_loss_local, device)
                    tmp_loss_world = distributed_mean(tmp_loss_local, device)
                    tmp_loss_reg_local = (reg_loss / max(tmp_tokens, 1))
                    tmp_loss_reg_world = distributed_mean(tmp_loss_reg_local, device)
                    if is_main_process():
                        avg_ppl = math.exp(avg_loss_world) if avg_loss_world < 20 else float("inf")
                        tmp_ppl = math.exp(tmp_loss_world) if tmp_loss_world < 20 else float("inf")
                        if writer is not None:
                            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                            writer.add_scalar("train/epoch_avg_loss", avg_loss_world, global_step)
                            writer.add_scalar("train/epoch_avg_ppl", avg_ppl, global_step)
                            writer.add_scalar("train/tmp_loss", tmp_loss_world, global_step)
                            writer.add_scalar("train/tmp_ppl", tmp_ppl, global_step)
                            writer.add_scalar("train/tmp_reg_loss", tmp_loss_reg_world, global_step)
                        if isinstance(pbar, tqdm):
                            pbar.set_postfix({"lr": lr_scheduler.get_last_lr()[0],
                                            "epoch_avg_loss": f"{avg_loss_world:.4f}", "epoch_avg_ppl": f"{avg_ppl:.2f}",
                                            "tmp_loss": f"{tmp_loss_world:.4f}", "tmp_ppl": f"{tmp_ppl:.2f}",
                                            "tmp_reg_loss": f"{tmp_loss_reg_world:.8f}"})
                    tmp_loss = 0.0
                    tmp_tokens = 0

                # ---- Periodic checkpoint (rank 0 only) ----
                if getattr(cfg.save, "save_steps", 0) and global_step % cfg.save.save_steps == 0:
                    if ddp_is_active():
                        dist.barrier()
                    if is_main_process():
                        ckpt_dir = os.path.join(ckpt_root, f"checkpoint-{global_step}")
                        logger.info(f"Saving checkpoint to {ckpt_dir}")
                        # Save unwrapped metanetwork (state is in ddp_metanet.module when DDP)
                        save_checkpoint(
                            ddp_metanet.module if isinstance(ddp_metanet, DDP) else ddp_metanet,
                            ckpt_dir,
                            extra_state={"global_step": global_step},
                            metalora=metalora,
                        )
                        save_training_state(
                            ckpt_dir,
                            global_step,
                            epoch,
                            step,
                            best_eval_loss,
                        )
                    if ddp_is_active():
                        dist.barrier()

                # ---- Eval + best checkpoint ----
                if getattr(cfg.eval, "eval_steps", 0) and global_step % cfg.eval.eval_steps == 0:
                    eval_metrics = evaluate(ddp_metanet, val_loader, device, use_amp=cfg.run.use_amp, metalora=metalora, amp_dtype=amp_dtype)
                    if writer is not None:
                        writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
                        writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
                    if is_main_process():
                        logger.info(f"[Eval @ step {global_step}] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")

                    # # Best checkpoint saving on rank 0
                    # if getattr(cfg.save, "save_best", True) and is_main_process():
                    #     if eval_metrics["eval_loss"] < best_eval_loss:
                    #         best_eval_loss = eval_metrics["eval_loss"]
                    #         best_dir = os.path.join(ckpt_root, "best")
                    #         logger.info(f"New best model! Saving to {best_dir}")
                    #         save_checkpoint(
                    #             ddp_metanet.module if isinstance(ddp_metanet, DDP) else ddp_metanet,
                    #             best_dir,
                    #             extra_state={"global_step": global_step, "best_eval_loss": best_eval_loss},
                    #         )
                    #         save_training_state(
                    #             best_dir,
                    #             global_step,
                    #             epoch,
                    #             step,
                    #             best_eval_loss,
                    #         )
                    # if ddp_is_active():
                    #     dist.barrier()
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Epoch-end eval/log (averaged)
        avg_epoch_loss_local = (epoch_loss / max(epoch_tokens, 1))
        avg_epoch_loss_world = distributed_mean(avg_epoch_loss_local, device)
        epoch_ppl = math.exp(avg_epoch_loss_world) if avg_epoch_loss_world < 20 else float("inf")
        if is_main_process():
            logger.info(f"Epoch {epoch} done. train_loss={avg_epoch_loss_world:.4f} train_ppl={epoch_ppl:.2f}")

        eval_metrics = evaluate(ddp_metanet, val_loader, device, use_amp=cfg.run.use_amp, metalora=metalora, amp_dtype=amp_dtype)
        if writer is not None:
            writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
            writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
        if is_main_process():
            logger.info(f"[Epoch {epoch} Eval] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")

        if getattr(cfg.save, "save_best", True) and is_main_process():
            if eval_metrics["eval_loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval_loss"]
                best_dir = os.path.join(ckpt_root, "best")
                logger.info(f"New best model! Saving to {best_dir}")
                save_checkpoint(
                    ddp_metanet.module if isinstance(ddp_metanet, DDP) else ddp_metanet,
                    best_dir,
                    extra_state={"global_step": global_step, "best_eval_loss": best_eval_loss},
                    metalora=metalora,
                )
                save_training_state(
                    best_dir,
                    global_step,
                    epoch,
                    step,
                    best_eval_loss,
                )
        if ddp_is_active():
            dist.barrier()
    
    # # Initial eval
    # if resume_dir is None:
    #     init_eval_without_metanetwork = evaluate(ddp_metanet, val_loader, device, use_amp=cfg.run.use_amp, use_metanet=False, amp_dtype=amp_dtype)
    #     if is_main_process():
    #         logger.info(f"[without lora] loss={init_eval_without_metanetwork['eval_loss']:.4f} ppl={init_eval_without_metanetwork['perplexity']:.2f}")
    # init_eval = evaluate(ddp_metanet, val_loader, device, use_amp=cfg.run.use_amp, metalora=metalora, amp_dtype=amp_dtype)
    # if writer is not None:
    #     writer.add_scalar("eval/loss", init_eval["eval_loss"], global_step)
    #     writer.add_scalar("eval/ppl", init_eval["perplexity"], global_step)
    # if is_main_process():
    #     logger.info(f"[Eval @ step {global_step}] loss={init_eval['eval_loss']:.4f} ppl={init_eval['perplexity']:.2f}")

    # Main training epochs
    for epoch in range(1, cfg.optim.num_epochs + 1):
        one_train_epoch(epoch, start_epoch, start_step_in_epoch)
        if cfg.data.source == "squad":
            train_ds.shuffle()
        

    # Final save (rank 0 only)
    if is_main_process():
        logger.info("Saving final model...")
        final_dir = os.path.join(ckpt_root, "final")
        save_checkpoint(
            ddp_metanet.module if isinstance(ddp_metanet, DDP) else ddp_metanet,
            final_dir,
            extra_state={"global_step": global_step},
            metalora=metalora,
        )

        if cfg.paths.output_dir:
            stable_out = cfg.paths.output_dir
            os.makedirs(stable_out, exist_ok=True)
            save_checkpoint(
                ddp_metanet.module if isinstance(ddp_metanet, DDP) else ddp_metanet,
                stable_out,
                extra_state={"global_step": global_step},
                metalora=metalora,
            )
            logger.info(f"Model saved to {stable_out}")

        logger.info(f"Complete !")

    if writer is not None:
        writer.close()

    # Cleanup DDP
    ddp_cleanup_if_needed()


if __name__ == "__main__":
    main()
