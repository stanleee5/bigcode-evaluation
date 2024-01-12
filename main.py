"""
LLM Code Evaluation
"""
import argparse
import asyncio
import copy
import os
import shutil
import time
from typing import Dict, List, Tuple

import orjson
import ray
import torch
from loguru import logger

from bigeval.generator import VllmGenerator
from bigeval.prompt_template import PROMPT_TEMPLATES
from bigeval.task import Task
from bigeval.tasks import ALL_TASKS

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    # Main Logic
    parser.add_argument("--tasks", type=str, default="humaneval")
    parser.add_argument("--generation-path", default="outputs/generations.json")
    parser.add_argument("--metric-path", default="outputs/metrics.json")
    parser.add_argument("--log-path", default="outputs/logs.json")
    parser.add_argument(
        "--save-dir", default=None, help="root directory to save generation/metric/log"
    )
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--load-saved", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="for debugging")

    # LLM
    parser.add_argument("-f", "--framework", type=str, default="vllm")
    parser.add_argument("-m", "--model")
    parser.add_argument("-a", "--adapter")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("-tp", "--tensor-parallel-size", default=1, type=int)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu-memory-utilization", default=0.95, type=float)
    parser.add_argument("--swap-space", default=16, type=int)
    parser.add_argument("--model-cache", default="./merged_models")

    # Generation
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--top-k", default=None, type=int)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--n-samples", default=1, type=int)
    parser.add_argument("--max-tokens", default=512, type=int)

    # Task
    parser.add_argument("--strip-prompt", default=True, type=bool)
    parser.add_argument("--instruction-template", default="[INST]\n")
    parser.add_argument("--response-template", default="\n[/INST]\n")
    parser.add_argument("--template", default=None)
    parser.add_argument("--num-workers", default=4, type=int)

    args = parser.parse_args()
    assert args.model or args.adapter

    for k, v in vars(args).items():
        v_str = f"'{v}'" if isinstance(v, str) else str(v)
        logger.info(f"{k} = {v_str}")

    return args


def get_tasks(args) -> Dict[str, Task]:
    if args.template is not None:
        assert not args.instruction_template and not args.response_template
        logger.info(f"Prompt template from {args.template = }")
        template_format = PROMPT_TEMPLATES[args.template]

        args.instruction_template = template_format["instruction"]
        args.response_template = template_format["response"]
        logger.info(f"instruction: {args.instruction_template}")
        logger.info(f"response: {args.response_template}")

    tasks = {}
    for task_name in args.tasks.split(","):
        assert task_name in ALL_TASKS, f"TASKS: {list(ALL_TASKS.keys())}"

        task_class: Task = ALL_TASKS[task_name]
        tasks[task_name] = task_class(
            strip_prompt=args.strip_prompt,
            instruction_template=args.instruction_template,
            response_template=args.response_template,
            num_workers=args.num_workers,
        )
    return tasks


def evaluate_task(task: Task, generations: List[List[str]]) -> Tuple[Dict, Dict]:
    """apply task postprocess, and then evaluate"""
    references = [task.get_reference(x) for x in task.get_dataset()]
    generations = [
        [task.postprocess_generation(g, i) for g in gs]
        for i, gs in enumerate(generations)
    ]
    results, logs = task.process_results(generations, references)
    return results, logs


def dict_to_str(x) -> str:
    return orjson.dumps(
        x,
        option=orjson.OPT_INDENT_2
        | orjson.OPT_NON_STR_KEYS
        | orjson.OPT_SERIALIZE_NUMPY,
    ).decode("utf-8")


async def main(args):
    tasks: Dict[str, Task] = get_tasks(args)

    args_dict = copy.deepcopy(vars(args))
    task_generations = {"config": args_dict}
    task_metrics = {"config": args_dict}
    task_logs = {"config": args_dict}

    if args.save_dir:
        args.generation_path = os.path.join(args.save_dir, "generations.json")
        args.metric_path = os.path.join(args.save_dir, "metrics.json")
        args.log_path = os.path.join(args.save_dir, "logs.json")
        os.makedirs(args.save_dir, exist_ok=True)

    # Load from saved-path if already exists
    if args.load_saved or args.evaluation_only:
        with open(args.generation_path) as f:
            task_generations.update(**orjson.loads(f.read()))
            logger.warning(f"Loading generations from: {args.generation_path}")
            logger.info(f">> loaded tasks: {list(task_generations.keys())}")

    if not args.evaluation_only:
        if args.framework == "vllm":
            generator = VllmGenerator(args)
        else:
            raise NotImplementedError(f"{args.framework}")

        for task_name, task in tasks.items():
            logger.info(f"Generation: {task_name = }")
            start_time = time.perf_counter()
            generations: List[List[str]] = await generator.generate_task(
                task, max_samples=args.max_samples
            )
            elapsed = time.perf_counter() - start_time

            task_generations[task_name] = generations
            task_generations[f"{task_name}-elapsed"] = elapsed

            with open(args.generation_path, "w", encoding="utf-8") as f:
                f.write(dict_to_str(task_generations))
                logger.info(f"generation saved at: {args.generation_path}")

        # Free GPU-memory / Ray workers
        generator.close()
        del generator
        torch.cuda.empty_cache()
        ray.shutdown()

    if not args.generation_only:
        for task_name, task in tasks.items():
            generations = task_generations[task_name]
            logger.warning(
                f"Evaluate: '{task_name}' - {len(generations)} tasks * {len(generations[0])} candidates"
            )

            start_time = time.perf_counter()
            metrics, logs = evaluate_task(task, generations)
            elapsed = time.perf_counter() - start_time

            task_metrics[task_name] = metrics
            task_logs[task_name] = logs
            logger.info(f"{metrics} - {elapsed = :.2f} sec.")

            with open(args.metric_path, "w", encoding="utf-8") as f:
                f.write(dict_to_str(task_metrics))
                logger.info(f">> evaluation metrics saved at: {args.metric_path}")
            with open(args.log_path, "w", encoding="utf-8") as f:
                f.write(dict_to_str(task_logs))
                logger.info(f">> evaluation logs saved at: {args.metric_path}")

    logger.info(dict_to_str(task_metrics))


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
