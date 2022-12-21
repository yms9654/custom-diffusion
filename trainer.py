from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import subprocess

import gradio as gr
import PIL.Image
import torch
import json

os.environ['PYTHONPATH'] = f'custom-diffusion:{os.getenv("PYTHONPATH", "")}'


def pad_image(image: PIL.Image.Image) -> PIL.Image.Image:
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = PIL.Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = PIL.Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image


class Trainer:
    def __init__(self):
        self.is_running = False
        self.is_running_message = 'Another training is in progress.'

        self.output_dir = pathlib.Path('results')
        self.instance_data_dir = self.output_dir / 'training_data'
        self.class_data_dir = self.output_dir / 'regularization_data'

    def check_if_running(self) -> dict:
        if self.is_running:
            return gr.update(value=self.is_running_message)
        else:
            return gr.update(value='No training is running.')

    def cleanup_dirs(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def prepare_dataset(self, concept_images_collection: list, concept_prompt_collection: list, class_prompt_collection: list, resolution: int) -> None:
        self.instance_data_dir.mkdir(parents=True)
        concepts_list = []

        for i in range(len(concept_images_collection)):
            concept_dir =  self.instance_data_dir /  f'{i}'
            class_dir = self.class_data_dir / f'{i}'
            concept_dir.mkdir(parents=True)
            concept_images = concept_images_collection[i]

            concepts_list.append(
                    {
                        "instance_prompt": concept_prompt_collection[i],
                        "class_prompt": class_prompt_collection[i],
                        "instance_data_dir": f'{concept_dir}',
                        "class_data_dir": f'{class_dir}'
                    }
                )

            for i, temp_path in enumerate(concept_images):
                image = PIL.Image.open(temp_path.name)
                image = pad_image(image)
                # image = image.resize((resolution, resolution))
                image = image.convert('RGB')
                out_path = concept_dir / f'{i:03d}.jpg'
                image.save(out_path, format='JPEG', quality=100)

        print(concepts_list)
        json.dump(concepts_list, open( f'{self.output_dir}/temp.json' , 'w') )

        
    def run(
        self,
        base_model: str,
        resolution_s: str,
        n_steps: int,
        learning_rate: float,
        train_text_encoder: bool,
        modifier_token: bool,
        gradient_accumulation: int,
        batch_size: int,
        use_8bit_adam: bool,
        gradient_checkpointing: bool,
        gen_images: bool,
        num_reg_images: int,
        *inputs, 
    ) -> tuple[dict, list[pathlib.Path]]:
        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')

        num_concept = 0
        for i in range(len(inputs) // 3):
            if inputs[i] != None:
                num_concept +=1

        print(num_concept, inputs)
        concept_images_collection = inputs[: num_concept]
        concept_prompt_collection = inputs[3:  3 + num_concept]
        class_prompt_collection = inputs[6: 6+num_concept]
        if self.is_running:
            return gr.update(value=self.is_running_message), []

        if concept_images_collection is None:
            raise gr.Error('You need to upload images.')
        if not concept_prompt_collection:
            raise gr.Error('The concept prompt is missing.')

        resolution = int(resolution_s)

        self.cleanup_dirs()
        self.prepare_dataset(concept_images_collection, concept_prompt_collection, class_prompt_collection, resolution)
        torch.cuda.empty_cache()
        command = f'''
        accelerate launch custom-diffusion/src/diffuser_training.py \
          --pretrained_model_name_or_path={base_model}   \
          --output_dir={self.output_dir} \
          --concepts_list={f'{self.output_dir}/temp.json'} \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --resolution={resolution}  \
          --train_batch_size={batch_size}  \
          --gradient_accumulation_steps={gradient_accumulation}  \
          --learning_rate={learning_rate}  \
          --lr_scheduler="constant" \
          --lr_warmup_steps=0 \
          --max_train_steps={n_steps} \
          --num_class_images={num_reg_images} \
          --initializer_token="ktn+pll+ucd" \
          --scale_lr --hflip 
        '''
        if modifier_token:
            tokens = '+'.join([f'<new{i+1}>' for i in range(num_concept)])
            command += f' --modifier_token {tokens}'
            
        if not gen_images:
            command += ' --real_prior'
        if use_8bit_adam:
            command += ' --use_8bit_adam'
        if train_text_encoder:
            command += f' --train_text_encoder'
        if gradient_checkpointing:
            command += f' --gradient_checkpointing'
        
        with open(self.output_dir / 'train.sh', 'w') as f:
            command_s = ' '.join(command.split())
            f.write(command_s)

        self.is_running = True
        res = subprocess.run(shlex.split(command))
        self.is_running = False

        if res.returncode == 0:
            result_message = 'Training Completed!'
        else:
            result_message = 'Training Failed!'
        weight_paths = sorted(self.output_dir.glob('*.bin'))
        return gr.update(value=result_message), weight_paths
