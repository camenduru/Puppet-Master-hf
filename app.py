import os
from PIL import Image, ImageOps
import random

import cv2
from diffusers import StableVideoDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import XFormersAttnProcessor
from diffusers.utils import export_to_gif
import gradio as gr
import numpy as np
from safetensors import safe_open
from segment_anything import build_sam, SamPredictor
import spaces
from tqdm import tqdm
import torch

from svd import (
    UNetDragSpatioTemporalConditionModel,
    AllToFirstXFormersAttnProcessor,
)


TITLE = '''Puppet-Master: Scaling Interactive Video Generation as a Motion Prior for Part-Level Dynamics'''
DESCRIPTION = """
<div>
Try <a href='https://vgg-puppetmaster.github.io/'><b>Puppet-Master</b></a> yourself to animate your favorite objects in seconds!
</div>
<div>
Please give us a 🌟 on <a href='https://github.com/RuiningLi/puppet-master'>Github</a> if you like our work!
</div>
"""
INSTRUCTION = '''
2 steps to get started:
- Upload an image of a dynamic object.
- Add one or more drags on the object to specify the part-level interactions.
How to add drags:
- To add a drag, first click on the starting point of the drag, then click on the ending point of the drag, on the Input Image (leftmost).
- You can add up to 5 drags.
- After every click, the drags will be visualized on the Image with Drags (second from left).
- If the last drag is not completed (you specified the starting point but not the ending point), it will simply be ignored.
- To retry, click the [x] button on the top-right corner of the input image to start over, even if you just want to try a different set of drags.
- Have fun dragging!

Then, you will be prompted to verify the object segmentation. Once you confirm that the segmentation is decent, the output image will be generated in seconds!

Tips:
- We found having classifier-free guidance weight ~5.0 works best.
- Try changing the random seed to get different results.
'''
PREPROCESS_INSTRUCTION = '''
Segmentation is needed if it is not already provided through an alpha channel in the input image.
You don't need to tick this box if you have chosen one of the example images.
If you have uploaded one of your own images, it is very likely that you will need to tick this box.
You should verify that the preprocessed image is object-centric (i.e., clearly contains a single object) and has white background.
'''


def tensor2vid(video: torch.Tensor, processor: VaeImageProcessor, output_type: str = "np"):
    batch_size = video.shape[0]
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


def center_and_square_image(pil_image_rgba, drags, scale_factor):
    image = pil_image_rgba
    alpha = np.array(image)[:, :, 3]  # Extract the alpha channel

    foreground_coords = np.argwhere(alpha > 0)
    y_min, x_min = foreground_coords.min(axis=0)
    y_max, x_max = foreground_coords.max(axis=0)
    cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
    crop_height, crop_width = y_max - y_min + 1, x_max - x_min + 1
    side_length = int(max(crop_height, crop_width) * scale_factor)
    padded_image = ImageOps.expand(
        image, 
        (side_length // 2, side_length // 2, side_length // 2, side_length // 2), 
        fill=(255, 255, 255, 255)
    )
    left, top = cx, cy
    new_drags = []
    for d in drags:
        x, y = d
        new_x, new_y = (x + side_length // 2 - cx) / side_length, (y + side_length // 2 - cy) / side_length
        new_drags.append((new_x, new_y))

    # Crop or pad the image as needed to make it centered around (cx, cy)
    image = padded_image.crop((left, top, left + side_length, top + side_length))
    # Resize the image to 256x256
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    return image, new_drags


def sam_init():
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "ckpts", "sam_vit_h_4b8939.pth")
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to("cuda"))
    return predictor


def model_init():
    model_checkpoint = os.path.join(os.path.dirname(__file__), "ckpts", "model.safetensors")
    state_dict = {}
    with safe_open(model_checkpoint, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    model = UNetDragSpatioTemporalConditionModel(num_drags=5)
    attn_processors_dict={
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "mid_block.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "mid_block.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "mid_block.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "mid_block.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
    }
    model.set_attn_processor(attn_processors_dict)
    model.load_state_dict(state_dict, strict=True)
    return model.to("cuda")


sam_predictor = sam_init()
model = model_init()
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "/scratch/shared/beegfs/ruining/projects/generative-models/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16, variant="fp16"
)
pipe.vae.to(dtype=torch.float16, device="cuda")
pipe.image_encoder = pipe.image_encoder.to("cuda")


@spaces.GPU(duration=10)
def sam_segment(input_image, drags, foreground_points=None, scale_factor=2.2):
    image = np.asarray(input_image)
    sam_predictor.set_image(image)

    with torch.no_grad():
        masks_bbox, _, _ = sam_predictor.predict(
            point_coords=foreground_points if foreground_points is not None else None,
            point_labels=np.ones(len(foreground_points)) if foreground_points is not None else None,
            multimask_output=True
        )

    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    out_image, new_drags = center_and_square_image(Image.fromarray(out_image, mode="RGBA"), drags, scale_factor)

    return out_image, new_drags


def get_point(img, sel_pix, evt: gr.SelectData):
    sel_pix.append(evt.index)
    points = []
    img = np.array(img)
    height = img.shape[0]
    arrow_width_large = 7 * height // 256
    arrow_width_small = 3 * height // 256
    circle_size = 5 * height // 256

    with_alpha = img.shape[2] == 4
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 1:
            cv2.circle(img, tuple(point), circle_size, (0, 0, 255, 255) if with_alpha else (0, 0, 255), -1)
        else:
            cv2.circle(img, tuple(point), circle_size, (255, 0, 0, 255) if with_alpha else (255, 0, 0), -1)
        points.append(tuple(point))
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (0, 0, 0, 255) if with_alpha else (0, 0, 0), arrow_width_large)
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 0, 255) if with_alpha else (0, 0, 0), arrow_width_small)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


def clear_drag():
    return []


def preprocess_image(img, chk_group, drags):
    if img is None:
        gr.Warning("No image is specified. Please specify an image before preprocessing.")
        return None, drags

    if drags is None or len(drags) == 0:
        foreground_points = None
    else:
        foreground_points = np.array([drags[i] for i in range(0, len(drags), 2)])

    if len(drags) == 0:
        gr.Warning("No drags are specified. We recommend first specifying the drags before preprocessing.")

    new_drags = drags
    if "Preprocess with Segmentation" in chk_group:
        img_np = np.array(img)
        rgb_img = img_np[..., :3]
        img, new_drags = sam_segment(
            rgb_img,
            drags,
            foreground_points=foreground_points,
        )
    else:
        new_drags = [(d[0] / img.width, d[1] / img.height) for d in drags]

    img = np.array(img).astype(np.float32)
    processed_img = img[..., :3] * img[..., 3:] / 255. + 255. * (1 - img[..., 3:] / 255.)
    image_pil = Image.fromarray(processed_img.astype(np.uint8), mode="RGB")
    processed_img = image_pil.resize((256, 256), Image.LANCZOS)
    return processed_img, new_drags


def sample_from_noise(model, scheduler, cond_latent, cond_embedding, drags,
                      min_guidance=1.0, max_guidance=3.0, num_inference_steps=50):
    model.eval()

    scheduler.set_timesteps(num_inference_steps, device=cond_latent.device)
    timesteps = scheduler.timesteps.to(cond_latent.device)

    do_classifier_free_guidance = max_guidance > 1.0
    latents = torch.randn((1, 14, 4, 32, 32)).to(cond_latent) * scheduler.init_noise_sigma
    guidance_scale = torch.linspace(min_guidance, max_guidance, 14).unsqueeze(0).to(cond_latent)[..., None, None, None]

    for i, t in tqdm(enumerate(timesteps)):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = model(
                latent_model_input,
                t,
                image_latents=torch.cat([cond_latent, torch.zeros_like(cond_latent)]) if do_classifier_free_guidance else cond_latent,
                encoder_hidden_states=torch.cat([cond_embedding, torch.zeros_like(cond_embedding)]) if do_classifier_free_guidance else cond_embedding,
                added_time_ids=None,  # dummy
                drags=torch.cat([drags, torch.zeros_like(drags)]) if do_classifier_free_guidance else drags,
            )

        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents


@spaces.GPU(duration=40)
def generate_image(img_cond, seed, cfg_scale, drags_list):
    if img_cond is None:
        gr.Warning("Please preprocess the image first.")
        return None
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    img_cond_pil = Image.fromarray(img_cond)
    img_cond_preprocessed = pipe.video_processor.preprocess(img_cond_pil, height=256, width=256)
    img_cond_preprocessed = img_cond_preprocessed.to(device="cuda", dtype=torch.float16)
    latent_dist = pipe.vae.encode(img_cond_preprocessed).latent_dist
    embeddings = pipe._encode_image(img_cond_pil, device="cuda", num_videos_per_prompt=1, do_classifier_free_guidance=False)

    drags = torch.zeros(14, 5, 4)
    for i in range(0, len(drags_list), 2):
        start_point, end_point = drags_list[i:i+2]
        drag_idx = i // 2
        drags[:, drag_idx, :2] = torch.Tensor(start_point)
        drags[0, drag_idx, 2:] = torch.Tensor(start_point)
        drags[-1, drag_idx, 2:] = torch.Tensor(end_point)

        if drag_idx == 4:
            break

    frame_indices = torch.arange(1, 13).unsqueeze(-1).unsqueeze(-1)
    t = frame_indices.float() / 13.0  # Normalize time to [0, 1]
    drags[1:-1, :, 2:] = drags[0, :, 2:] * (1 - t) + drags[-1, :, 2:] * t
    drags = drags[None].to(device="cuda")

    batch = dict(
        drags=drags,
        cond_embedding=embeddings.to(dtype=torch.float32),
        cond_latent=latent_dist.mean.to(dtype=torch.float32),
    )

    with torch.no_grad():
        latents = sample_from_noise(
            model,
            pipe.scheduler,
            **batch,
            max_guidance=cfg_scale,
            num_inference_steps=50,
        )

        frames = pipe.vae.decode(latents.flatten(0, 1).to(torch.float16) / 0.18215, num_frames=14).sample.float()
        frames = tensor2vid(frames.view(-1, 14, 3, 256, 256).permute(0, 2, 1, 3, 4), pipe.video_processor, output_type="pil")[0]

    # Add drags
    frame_with_drag = np.ascontiguousarray(np.array(frames[0]))
    for i in range(0, len(drags_list), 2):
        drag_idx = i // 2
        start_point, end_point = drags_list[i:i+2]
        start_point = (int(start_point[0] * 256), int(start_point[1] * 256))
        end_point = (int(end_point[0] * 256), int(end_point[1] * 256))
        frame_with_drag = cv2.arrowedLine(frame_with_drag, start_point, end_point, (0, 0, 0), 4)
        frame_with_drag = cv2.arrowedLine(frame_with_drag, start_point, end_point, (255, 255, 0), 2)

        if drag_idx == 4:
            break

    frames = [Image.fromarray(frame_with_drag)] * 5 + frames
    save_dir = os.path.join(os.path.dirname(__file__), "outputs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_id = len(os.listdir(save_dir))
    save_path = os.path.join(save_dir, f"{save_id:05d}.gif")
    export_to_gif(frames, save_path)
    return save_path


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown("# " + DESCRIPTION)

    with gr.Row():
        gr.Markdown(INSTRUCTION)
    
    drags = gr.State(value=[])

    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            input_image = gr.Image(
                interactive=True,
                type='pil',
                image_mode="RGBA",
                width=256,
                show_label=True,
                label="Input Image",
            )

            example_folder = os.path.join(os.path.dirname(__file__), "./example_images")
            example_fns = [os.path.join(example_folder, example) for example in sorted(os.listdir(example_folder))]
            gr.Examples(
                examples=example_fns,
                inputs=[input_image],
                cache_examples=False,
                label='Feel free to use one of our provided examples!',
                examples_per_page=30
            )

            input_image.change(
                fn=clear_drag,
                outputs=[drags],
            )

        with gr.Column(scale=1):
            drag_image = gr.Image(
                type="numpy",
                label="Image with Drags",
                interactive=False,
                width=256,
                image_mode="RGB",
            )

            input_image.select(
                fn=get_point,
                inputs=[input_image, drags],
                outputs=[drag_image],
            )
        
        with gr.Column(scale=1):
            processed_image = gr.Image(
                type='numpy', 
                label="Processed Image", 
                interactive=False, 
                width=256,
                height=256,
                image_mode='RGB',
            )
            processed_image_highres = gr.Image(type='pil', image_mode='RGB', visible=False)

            with gr.Accordion('Advanced preprocessing options', open=True):
                with gr.Row():
                    with gr.Column():
                        preprocess_chk_group = gr.CheckboxGroup(
                            ['Preprocess with Segmentation'], 
                            label='Segment',
                            info=PREPROCESS_INSTRUCTION
                        )
            
            preprocess_button = gr.Button(
                value="Preprocess Input Image",
            )
            preprocess_button.click(
                fn=preprocess_image,
                inputs=[input_image, preprocess_chk_group, drags],
                outputs=[processed_image, drags],
                queue=True,
            )

        with gr.Column(scale=1):
            generated_gif = gr.Image(
                type="filepath",
                label="Generated GIF",
                interactive=False,
                height=256,
                width=256,
                image_mode="RGB",
            )

            with gr.Accordion('Advanced generation options', open=True):
                with gr.Row():
                    with gr.Column():
                        seed = gr.Slider(label="seed", value=0, minimum=0, maximum=10000, step=1, randomize=False)
                        cfg_scale = gr.Slider(
                            label="classifier-free guidance weight",
                            value=5, minimum=1, maximum=10, step=0.1
                        )

            generate_button = gr.Button(
                value="Generate Image",
            )
            generate_button.click(
                fn=generate_image,
                inputs=[processed_image, seed, cfg_scale, drags],
                outputs=[generated_gif],
            )

    demo.launch(share=True)
