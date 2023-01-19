from __future__ import annotations

import gc
import pathlib
import sys

import gradio as gr
import PIL.Image
import numpy as np

import torch
from diffusers import StableDiffusionPipeline
# from diffusers import EulerAncestralDiscreteScheduler
sys.path.insert(0, './custom-diffusion')


class InferencePipeline:
    def __init__(self):
        self.pipe = None
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weight_path = None

    def clear(self) -> None:
        self.weight_path = None
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def get_weight_path(name: str) -> pathlib.Path:
        curr_dir = pathlib.Path(__file__).parent
        return curr_dir / name

    def load_pipe(self, model_id: str, filename: str) -> None:
        weight_path = self.get_weight_path(filename)
        if weight_path == self.weight_path:
            return
        self.weight_path = weight_path
        weight = torch.load(self.weight_path, map_location=self.device)

        if self.device.type == 'cpu':
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16)
            pipe = pipe.to(self.device)

        from src import diffuser_training
        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, weight_path, compress=False)

        self.pipe = pipe

    def run(
        self,
        base_model: str,
        weight_name: str,
        prompt: str,
        seed: int,
        n_steps: int,
        guidance_scale: float,
        eta: float,
        batch_size: int,
        resolution: int,
    ) -> PIL.Image.Image:
        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')

        self.load_pipe(base_model, weight_name)

        if seed == -1:
            import random
            seed = random.randint(1, 100000)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        ret = []
        for i in range(2):
            out = self.pipe([prompt]*batch_size,
                            num_inference_steps=n_steps,
                            guidance_scale=guidance_scale,
                            height=resolution, width=resolution,
                            eta = eta,
                            generator=generator)  # type: ignore

            out = out.images
            for x in out:
                ret.append(PIL.Image.fromarray(np.array(x)))

        # torch.cuda.empty_cache()
        return ret
