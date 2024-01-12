import asyncio
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List

import requests
from loguru import logger
from tqdm.asyncio import tqdm

from bigeval.merge_peft import merge_peft_and_save
from bigeval.task import Task


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
