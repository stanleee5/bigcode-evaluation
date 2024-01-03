# bigcode-evaluation
Simple [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) using [vllm](https://github.com/vllm-project/vllm)

## Usage
```bash
docker build -t dice/bigcode-evaluation .
```

```bash
export MODEL_DIR=codellama/CodeLlama-7b-hf
export TASKS="humaneval,humanevalsynthesize-python,humanevalsynthesize-cpp"
export SAVE_DIR="outputs/codellama-7b"

# in docker
python3 main.py -m $MODEL_DIR --tasks $TASKS --save-dir $SAVE_DIR

# out docker
docker run -it --shm-size 4g --rm \
    --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
    -v $SAVE_DIR:/app/$SAVE_DIR \
    dice/bigcode-evaluation \
    -m $MODEL_DIR --tasks $TASKS --save-dir $SAVE_DIR

```

### Issue
- works on TP=1 only
