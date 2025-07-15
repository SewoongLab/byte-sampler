# Byte sampling

This is the implementation of [Sampling from Your Language Model One Byte at a Time](https://arxiv.org/abs/2506.14123). 

## Installation 

Clone the repository. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and then do:

```bash
uv run ipython
```

You can access this project's functionality via the `byte_sampler` module.

## Usage

You can then sample from a model byte-wise:

```python
from byte_sampling import *

# load in a model
bc = ByteConditioning("meta-llama/Llama-3.1-8B")

# sample a continuation with a QA formatted prompt.
generate_batched(
    BytewiseQAFactory(bc),
    ["What is your favorite flavor of ice cream?"],
    stop_strings=('\n',),
    display=True
)
```

