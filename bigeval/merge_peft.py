"""
Merge the PEFT model and save.
"""
import argparse
import os

import torch
from loguru import logger
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def merge_peft_and_save(model_dir, dtype, save_dir=None):
    torch_dtype = torch.float16
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    logger.info("Loading the model it might take a while without feedback")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        revision=None,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    logger.info(f"Merging the PEFT weights.")

    base_model_dir = model.peft_config["default"].base_model_name_or_path

    model = model.merge_and_unload()

    save_dir = model_dir if save_dir is None else save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving the newly created merged model to {save_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.save_pretrained(save_dir)

    model.save_pretrained(save_dir, safe_serialization=True)
    model.config.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    download_and_unload_peft(args.model, args.dtype, args.save_dir)
