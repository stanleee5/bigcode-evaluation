from abc import ABC, abstractmethod
from typing import Dict, List

from loguru import logger
from vllm import LLM, SamplingParams

from .task import Task


class VllmGenerator:
    def __init__(self, args):
        self.llm = LLM(
            model=args.model,
            dtype=args.dtype,
            quantization=args.quantization,
            gpu_memory_utilization=args.gpu_memory_utilization,
            swap_space=args.swap_space,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        self.sampling_params = SamplingParams(
            temperature=0 if not args.do_sample else args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            n=args.n_samples,
            max_tokens=args.max_tokens,
        )
        logger.info(f"Generator Initialized")

    def generate_task(self, task: Task, max_samples: int = None) -> List[List[str]]:
        """
        Return full generation(given_prompt + response) for task.
        Do not apply task post-processing here.
        """
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
