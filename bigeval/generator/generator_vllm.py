import os
import shutil
import uuid
from typing import List

from loguru import logger
from vllm import LLM, SamplingParams

from ..merge_peft import merge_peft_and_save
from .generator import Generator


class VllmGenerator(Generator):
    def __init__(self, args):
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
            max_model_len=args.max_model_len,
        )
        self.sampling_params = SamplingParams(
            temperature=0 if not args.do_sample else args.temperature,
            top_k=args.top_k if args.top_k else -1,
            top_p=args.top_p if args.top_p else None,
            n=args.n_samples,
            max_tokens=args.max_tokens,
        )
        logger.info(f"vllm Generator Initialized")

    def inference(self, prompts: List[str]) -> List[List[str]]:
        req_outputs = self.llm.generate(prompts, self.sampling_params)
        generations = [[o.text for o in output.outputs] for output in req_outputs]
        return generations

    def close(self):
        if self.model_cache_dir:
            logger.info(f"Remove the PEFT-merged cache: {self.model_cache_dir}")
            shutil.rmtree(self.model_cache_dir)
        super().close()


if __name__ == "__main__":
    """
    Test vllm Inference
    - usage: python -m bigeval.generator.generator_vllm
    """

    class Args:
        model = "JackFram/llama-160m"
        max_model_len = None
        adapter = None
        n_samples = 2
        do_sample = True
        max_tokens = 8
        temperature = 0.2
        top_k = 50
        top_p = 0.95
        batch_size = 2
        dtype = "float16"
        tensor_parallel_size = 1
        quantization = None
        gpu_memory_utilization = 0.9
        swap_space = 16

    prompts = ["My name is", "Life is"]

    generations_list = VllmGenerator(Args()).inference(prompts)
    for generations in generations_list:
        logger.info(generations)
