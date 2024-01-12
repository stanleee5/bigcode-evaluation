# bigcode-evaluation
Simple [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) using llm-inference frameworks

## Usage
```bash
docker build -t dice/bigcode-evaluation .
```

```bash
export MODEL_DIR=codellama/CodeLlama-7b-Instruct-hf
export TASKS="humaneval,humanevalsynthesize-python,humanevalsynthesize-cpp"
export SAVE_DIR="outputs/CodeLlama-7b-Instruct"

# in docker
python3 main.py -m $MODEL_DIR --tasks $TASKS --save-dir $SAVE_DIR \
    --instruction-template $'[INST] ' \
    --response-template $' [/INST] '

# out docker
docker run -it --shm-size 4g --rm \
    --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
    -v $SAVE_DIR:/app/$SAVE_DIR \
    dice/bigcode-evaluation \
    # params for entrypint (python3 main.py)
```
