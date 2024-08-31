

# Stable Diffusion Image Generation

This project uses the Stable Diffusion model to generate images from text prompts.

## Setup

1. Install the required dependencies:

```
pip install diffusers transformers accelerate
```

2. Import the necessary libraries:

```python
import torch
from diffusers import StableDiffusionPipeline
```

3. Load the Stable Diffusion model:

```python
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
```

## Usage

To generate an image from a text prompt:

```python
prompt = "Your text prompt here"
image = pipe(prompt).images[0]
image.save("generated_image.png")
```

## Features

- Generates high-quality images from text descriptions
- Uses the Stable Diffusion v1.5 model
- Supports GPU acceleration for faster image generation

## Requirements

- Python 3.6+
- CUDA-capable GPU (for optimal performance)
- PyTorch
- Diffusers library
- Transformers library
- Accelerate library

## Notes

- The model downloads may take some time on first run
- Image generation quality and speed depend on your hardware capabilities

