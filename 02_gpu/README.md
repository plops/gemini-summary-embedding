

install cuml with `uv pip install cuml-cu12 cudf-cu12`

perform `uv sync` to build the local environment

run `CUDA_PATH=/opt/cuda/ uv run main.py`

start an interactive python shell using
```
CUDA_PATH=/opt/cuda/ uv run python -i main.py 
```