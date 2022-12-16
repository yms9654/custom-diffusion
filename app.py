#!/usr/bin/env python
"""Unofficial demo app for https://github.com/adobe-research/custom-diffusion.

The code in this repo is partly adapted from the following repository:
https://huggingface.co/spaces/hysts/LoRA-SD-training
The license of the original code is MIT, which is specified in the README.md.
"""

from __future__ import annotations

import os
import pathlib

import gradio as gr
import torch

from inference import InferencePipeline
from trainer import Trainer
from uploader import upload

TITLE = '# Custom Diffusion + StableDiffusion Training UI'
DESCRIPTION = 'This is an unofficial demo for [https://github.com/adobe-research/custom-diffusion](https://github.com/adobe-research/custom-diffusion).'

ORIGINAL_SPACE_ID = 'nupurkmr9/custom-diffusion'
SPACE_ID = os.getenv('SPACE_ID', ORIGINAL_SPACE_ID)
SHARED_UI_WARNING = f'''# Attention - This Space doesn't work in this shared UI. You can duplicate and use it with a paid private T4 GPU.

<center><a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></center>
'''
if os.getenv('SYSTEM') == 'spaces' and SPACE_ID != ORIGINAL_SPACE_ID:
    SETTINGS = f'<a href="https://huggingface.co/spaces/{SPACE_ID}/settings">Settings</a>'

else:
    SETTINGS = 'Settings'
CUDA_NOT_AVAILABLE_WARNING = f'''# Attention - Running on CPU.
<center>
You can assign a GPU in the {SETTINGS} tab if you are running this on HF Spaces.
"T4 small" is sufficient to run this demo.
</center>
'''


def show_warning(warning_text: str) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Box():
            gr.Markdown(warning_text)
    return demo


def update_output_files() -> dict:
    paths = sorted(pathlib.Path('results').glob('*.pt'))
    paths = [path.as_posix() for path in paths]  # type: ignore
    return gr.update(value=paths or None)


def create_training_demo(trainer: Trainer,
                         pipe: InferencePipeline) -> gr.Blocks:
    with gr.Blocks() as demo:
        base_model = gr.Dropdown(
            choices=['stabilityai/stable-diffusion-2-1-base'],
            value='stabilityai/stable-diffusion-2-1-base',
            label='Base Model',
            visible=False)
        resolution = gr.Dropdown(choices=['512'],
                                 value='512',
                                 label='Resolution',
                                 visible=False)

        with gr.Row():
            with gr.Box():
                gr.Markdown('Training Data')
                concept_images = gr.Files(label='Images for your concept')
                concept_prompt = gr.Textbox(label='Concept Prompt',
                                            max_lines=1)
                gr.Markdown('''
                    - Upload images of the style you are planning on training on.
                    - For a concept prompt, use a unique, made up word to avoid collisions.
                    ''')
            with gr.Box():
                gr.Markdown('Training Parameters')
                num_training_steps = gr.Number(
                    label='Number of Training Steps', value=1000, precision=0)
                learning_rate = gr.Number(label='Learning Rate', value=0.0001)
                train_text_encoder = gr.Checkbox(label='Train Text Encoder',
                                                 value=True)
                learning_rate_text = gr.Number(
                    label='Learning Rate for Text Encoder', value=0.00005)
                gradient_accumulation = gr.Number(
                    label='Number of Gradient Accumulation',
                    value=1,
                    precision=0)
                fp16 = gr.Checkbox(label='FP16', value=True)
                use_8bit_adam = gr.Checkbox(label='Use 8bit Adam', value=True)
                gr.Markdown('''
                    - It will take about 8 minutes to train for 1000 steps with a T4 GPU.
                    - You may want to try a small number of steps first, like 1, to see if everything works fine in your environment.
                    - Note that your trained models will be deleted when the second training is started. You can upload your trained model in the "Upload" tab.
                    ''')

        run_button = gr.Button('Start Training')
        with gr.Box():
            with gr.Row():
                check_status_button = gr.Button('Check Training Status')
                with gr.Column():
                    with gr.Box():
                        gr.Markdown('Message')
                        training_status = gr.Markdown()
                    output_files = gr.Files(label='Trained Weight Files')

        run_button.click(fn=pipe.clear)
        run_button.click(fn=trainer.run,
                         inputs=[
                             base_model,
                             resolution,
                             concept_images,
                             concept_prompt,
                             num_training_steps,
                             learning_rate,
                             train_text_encoder,
                             learning_rate_text,
                             gradient_accumulation,
                             fp16,
                             use_8bit_adam,
                         ],
                         outputs=[
                             training_status,
                             output_files,
                         ],
                         queue=False)
        check_status_button.click(fn=trainer.check_if_running,
                                  inputs=None,
                                  outputs=training_status,
                                  queue=False)
        check_status_button.click(fn=update_output_files,
                                  inputs=None,
                                  outputs=output_files,
                                  queue=False)
    return demo


def find_weight_files() -> list[str]:
    curr_dir = pathlib.Path(__file__).parent
    paths = sorted(curr_dir.rglob('*.pt'))
    paths = [path for path in paths if not path.stem.endswith('.text_encoder')]
    return [path.relative_to(curr_dir).as_posix() for path in paths]


def reload_lora_weight_list() -> dict:
    return gr.update(choices=find_weight_files())


def create_inference_demo(pipe: InferencePipeline) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                base_model = gr.Dropdown(
                    choices=['stabilityai/stable-diffusion-2-1-base'],
                    value='stabilityai/stable-diffusion-2-1-base',
                    label='Base Model',
                    visible=False)
                reload_button = gr.Button('Reload Weight List')
                lora_weight_name = gr.Dropdown(choices=find_weight_files(),
                                               value='lora/lora_disney.pt',
                                               label='LoRA Weight File')
                prompt = gr.Textbox(
                    label='Prompt',
                    max_lines=1,
                    placeholder='Example: "style of sks, baby lion"')
                alpha = gr.Slider(label='Alpha',
                                  minimum=0,
                                  maximum=2,
                                  step=0.05,
                                  value=1)
                alpha_for_text = gr.Slider(label='Alpha for Text Encoder',
                                           minimum=0,
                                           maximum=2,
                                           step=0.05,
                                           value=1)
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=100000,
                                 step=1,
                                 value=1)
                with gr.Accordion('Other Parameters', open=False):
                    num_steps = gr.Slider(label='Number of Steps',
                                          minimum=0,
                                          maximum=100,
                                          step=1,
                                          value=50)
                    guidance_scale = gr.Slider(label='CFG Scale',
                                               minimum=0,
                                               maximum=50,
                                               step=0.1,
                                               value=7)

                run_button = gr.Button('Generate')

                gr.Markdown('''
                - Models with names starting with "lora/" are the pretrained models provided in the [original repo](https://github.com/cloneofsimo/lora), and the ones with names starting with "results/" are your trained models.
                - After training, you can press "Reload Weight List" button to load your trained model names.
                - The pretrained models for "disney", "illust" and "pop" are trained with the concept prompt "style of sks".
                - The pretrained model for "kiriko" is trained with the concept prompt "game character bnha". For this model, the text encoder is also trained.
                ''')
            with gr.Column():
                result = gr.Image(label='Result')

        reload_button.click(fn=reload_lora_weight_list,
                            inputs=None,
                            outputs=lora_weight_name)
        prompt.submit(fn=pipe.run,
                      inputs=[
                          base_model,
                          lora_weight_name,
                          prompt,
                          alpha,
                          alpha_for_text,
                          seed,
                          num_steps,
                          guidance_scale,
                      ],
                      outputs=result,
                      queue=False)
        run_button.click(fn=pipe.run,
                         inputs=[
                             base_model,
                             lora_weight_name,
                             prompt,
                             alpha,
                             alpha_for_text,
                             seed,
                             num_steps,
                             guidance_scale,
                         ],
                         outputs=result,
                         queue=False)
    return demo


def create_upload_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        model_name = gr.Textbox(label='Model Name')
        hf_token = gr.Textbox(
            label='Hugging Face Token (with write permission)')
        upload_button = gr.Button('Upload')
        with gr.Box():
            gr.Markdown('Message')
            result = gr.Markdown()
        gr.Markdown('''
            - You can upload your trained model to your private Model repo (i.e. https://huggingface.co/{your_username}/{model_name}).
            - You can find your Hugging Face token [here](https://huggingface.co/settings/tokens).
            ''')

    upload_button.click(fn=upload,
                        inputs=[model_name, hf_token],
                        outputs=result)

    return demo


pipe = InferencePipeline()
trainer = Trainer()

with gr.Blocks(css='style.css') as demo:
    if os.getenv('IS_SHARED_UI'):
        show_warning(SHARED_UI_WARNING)
    if not torch.cuda.is_available():
        show_warning(CUDA_NOT_AVAILABLE_WARNING)

    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.TabItem('Train'):
            create_training_demo(trainer, pipe)
        with gr.TabItem('Test'):
            create_inference_demo(pipe)
        with gr.TabItem('Upload'):
            create_upload_demo()

demo.queue(default_enabled=False).launch(share=False)
