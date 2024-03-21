import asyncio
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List
from urllib.parse import urljoin

import requests
import torch
import tqdm
import transformers
from loguru import logger
from transformers import AutoTokenizer

from .merge_peft import merge_peft_and_save
from .task import Task

torch_dtypes = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class Generator(ABC):
    @abstractmethod
    async def generate_task(
        self, task: Task, max_samples: int = None
    ) -> List[List[str]]:
        """
        Return full generation(given_prompt + response) for task.
        Do not apply task post-processing here.
        """
        pass

    def close(self):
        logger.info(f"Close {self.__class__.__name__}")


class HFGenerator(Generator):
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=args.model,
            torch_dtype=torch_dtypes[args.dtype],
            device_map="auto",
            trust_remote_code=True,
        )

    async def generate_task(
        self, task: Task, max_samples: int = None
    ) -> List[List[str]]:
        logger.info("start HF pipeline")

        prompts: List[str] = [task.get_prompt(x) for x in task.get_dataset()]

        do_sample = self.args.do_sample
        bsz = self.args.batch_size

        responses = []
        for idx in tqdm.tqdm(range(0, len(prompts), bsz)):
            b_responses = self.pipeline(
                prompts[idx : idx + bsz],
                num_return_sequences=self.args.n_samples,
                do_sample=do_sample,
                temperature=self.args.temperature if do_sample else None,
                top_p=self.args.top_p if do_sample else None,
                top_k=self.args.top_k if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.args.max_tokens,
                return_full_text=False,
            )
            responses.extend(b_responses)

        generations: List[List[str]] = [
            [prompts[i] + r["generated_text"] for r in response]
            for i, response in enumerate(responses)
        ]
        return generations


class VllmGenerator(Generator):
    def __init__(self, args):
        from vllm import LLM, SamplingParams

        self.model_cache_dir = None
        if args.adapter:
            # save peft model to random cahce dir
            model_cache_dir = os.path.join(args.model_cache, str(uuid.uuid4())[:16])
            logger.info(f"Merge the PEFT model and save to: {model_cache_dir}")
            merge_peft_and_save(args.adapter, args.dtype, model_cache_dir)
            self.model_cache_dir = model_cache_dir

        self.llm = LLM(
            model=self.model_cache_dir if self.model_cache_dir else args.model,
            dtype=args.dtype,
            quantization=args.quantization,
            gpu_memory_utilization=args.gpu_memory_utilization,
            swap_space=args.swap_space,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=4096,
        )
        self.sampling_params = SamplingParams(
            temperature=0 if not args.do_sample else args.temperature,
            top_k=args.top_k if args.top_k else -1,
            top_p=args.top_p if args.top_p else None,
            n=args.n_samples,
            max_tokens=args.max_tokens,
        )
        logger.info(f"Generator Initialized")

    async def generate_task(
        self, task: Task, max_samples: int = None
    ) -> List[List[str]]:
        prompts: List[str] = [task.get_prompt(x) for x in task.get_dataset()]
        if max_samples:
            logger.warning(f"use small samples: {max_samples}")
            prompts = prompts[:max_samples]

        # List[vllm.outputs.RequestOutput]
        req_outputs = self.llm.generate(prompts, self.sampling_params)
        generations: List[List[str]] = [
            [prompts[i] + o.text for o in output.outputs]
            for i, output in enumerate(req_outputs)
        ]
        return generations

    def close(self):
        if self.model_cache_dir:
            logger.info(f"Remove the PEFT-merged cache: {self.model_cache_dir}")
            shutil.rmtree(self.model_cache_dir)
