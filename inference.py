from __future__ import annotations

import gc
import pathlib
import sys

import gradio as gr
import PIL.Image
import torch
from diffusers import StableDiffusionPipeline

sys.path.insert(0, 'lora')
from lora_diffusion import monkeypatch_lora, tune_lora_scale


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
    def get_lora_weight_path(name: str) -> pathlib.Path:
        curr_dir = pathlib.Path(__file__).parent
        return curr_dir / name

    @staticmethod
    def get_lora_text_encoder_weight_path(path: pathlib.Path) -> str:
        parent_dir = path.parent
        stem = path.stem
        text_encoder_filename = f'{stem}.text_encoder.pt'
        path = parent_dir / text_encoder_filename
        return path.as_posix() if path.exists() else ''

    def load_pipe(self, model_id: str, lora_filename: str) -> None:
        weight_path = self.get_lora_weight_path(lora_filename)
        if weight_path == self.weight_path:
            return
        self.weight_path = weight_path
        lora_weight = torch.load(self.weight_path, map_location=self.device)

        if self.device.type == 'cpu':
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16)
            pipe = pipe.to(self.device)

        monkeypatch_lora(pipe.unet, lora_weight)

        lora_text_encoder_weight_path = self.get_lora_text_encoder_weight_path(
            weight_path)
        if lora_text_encoder_weight_path:
            lora_text_encoder_weight = torch.load(
                lora_text_encoder_weight_path, map_location=self.device)
            monkeypatch_lora(pipe.text_encoder,
                             lora_text_encoder_weight,
                             target_replace_module=['CLIPAttention'])

        self.pipe = pipe

    def run(
        self,
        base_model: str,
        lora_weight_name: str,
        prompt: str,
        alpha: float,
        alpha_for_text: float,
        seed: int,
        n_steps: int,
        guidance_scale: float,
    ) -> PIL.Image.Image:
        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')

        self.load_pipe(base_model, lora_weight_name)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        tune_lora_scale(self.pipe.unet, alpha)  # type: ignore
        tune_lora_scale(self.pipe.text_encoder, alpha_for_text)  # type: ignore
        out = self.pipe(prompt,
                        num_inference_steps=n_steps,
                        guidance_scale=guidance_scale,
                        generator=generator)  # type: ignore
        return out.images[0]
