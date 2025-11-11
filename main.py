# main.py
import os
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from bqer import (
    calibrate_bqer,
    apply_bqer,
)
from evaluator import eval_ppl

# -------------------- ONE-CALL PIPELINE --------------------------------------

@torch.no_grad()
def calibrate_and_wrap_bqer(
    fp_model: nn.Module,
    q_model: nn.Module,
    dataloader,
    device: str = "cuda",
    args: argparse.Namespace = None,
):
    Ks_cur, Ks_prev = calibrate_bqer(
        fp_model, q_model, dataloader, device=device,
        lambda1=args.lambda1, clip=args.clip, group_size=args.group_size,
    )
    apply_bqer(q_model, Ks_cur, Ks_prev,
        window_m=args.window_m, alpha=args.alpha, group_size=args.group_size)
    return q_model

# -------------------- DATA LOADING -----------------------------------------

def prepare_dataloader(encodings, max_seqlen=16384, num_chunks=256):
    input_ids = encodings["input_ids"][0]
    attention_mask = encodings["attention_mask"][0] if "attention_mask" in encodings else torch.ones_like(input_ids)
    chunks = []
    for i in range(0, input_ids.shape[0] - max_seqlen + 1, max_seqlen):
        chunk_input_ids = input_ids[i:i+max_seqlen].unsqueeze(0)
        chunk_attention_mask = attention_mask[i:i+max_seqlen].unsqueeze(0)
        chunks.append({"input_ids": chunk_input_ids, "attention_mask": chunk_attention_mask})
    return chunks[:num_chunks]

# -------------------- EVALUATION -----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model", "-m",type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--qbits", "-q", type=int, default=2)
    # calibration data
    parser.add_argument("--seqlen", "-s", type=int, default=16384)
    parser.add_argument("--num_chunks", "-n", type=int, default=256)
    # BQER parameters
    parser.add_argument("--lambda1", "-l", type=float, default=1e-5)
    parser.add_argument("--window_m", "-w", type=int, default=0)
    parser.add_argument("--clip", "-c", type=float, default=0.25)
    parser.add_argument("--alpha", "-a", type=float, default=0.75)
    parser.add_argument("--group_size", "-g", type=int, default=512)
    args = parser.parse_args()

    HF_HOME = os.getenv("HF_HOME", "/data/hf_cache")
    # Load models
    fp_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    q_model = AutoModelForCausalLM.from_pretrained(f"{HF_HOME}/hub/{args.model.split('/')[-1]}-{args.qbits}bits-g64", 
                                                    dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp_model.config.use_cache = False
    q_model.config.use_cache = False

    dataset = load_dataset("THUDM/LongBench", "narrativeqa", split="test")
    encodings = tokenizer("\n\n".join([x["context"] for x in dataset if len(x["context"].strip()) > 0]), return_tensors="pt")
    dataloader = prepare_dataloader(encodings, max_seqlen=args.seqlen, num_chunks=args.num_chunks)

    print(f"Given config: " +
          f"[lambda1={args.lambda1}, window_m={args.window_m}, clip={args.clip}, alpha={args.alpha}, group_size={args.group_size}]")

    print(f"Evaluating perplexity of original model")
    ppl_test = eval_ppl(fp_model, tokenizer)
    print(f"wikitext perplexity of original model: {ppl_test}")

    print(f"Evaluating perplexity of quantized model")
    ppl_test = eval_ppl(q_model, tokenizer)
    print(f"wikitext perplexity of quantized model: {ppl_test}")

    print(f"Calibrating and wrapping BQER...")
    q_model = calibrate_and_wrap_bqer(
        fp_model, q_model, dataloader, device="cuda", args=args)

    del fp_model
    torch.cuda.empty_cache()

    print(f"Evaluating perplexity of BQER wrapped model")
    ppl_test = eval_ppl(q_model, tokenizer)
    print(f"wikitext perplexity of BQER wrapped model: {ppl_test}")
