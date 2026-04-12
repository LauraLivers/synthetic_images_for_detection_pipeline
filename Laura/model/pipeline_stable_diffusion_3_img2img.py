"""
A self-contained img2img pipeline for Stable Diffusion 3.5, adapted from
the original StableDiffusionXLImg2ImgPipeline with base_prompt conditioning.

The base_prompt trick:
- base_prompt (e.g. "a photo of a ladder") is encoded through CLIP-L + CLIP-G only,
  producing pooled_prompt_embeds that anchor object-class identity.
- The location prompt is encoded through all three encoders (CLIP-L, CLIP-G, T5),
  with T5 driving scene semantics and CLIP token embeddings driving style/texture.
- This decouples "what object" (pooled, from base_prompt) from "what scene" (T5 tokens).

Usage in generate_sd3.py:
    from pipeline_sd3_img2img import StableDiffusion3Img2ImgPipelineWithBasePrompt

    pipe = StableDiffusion3Img2ImgPipelineWithBasePrompt.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16,
    ).to(device)

    result = pipe(
        prompt="a 8k real photo: a ladder in a park",
        base_prompt="a photo of a ladder",
        image=pil_image,
        strength=0.5,
        guidance_scale=7.0,
        num_inference_steps=28,
    )
"""

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import PIL.Image

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models import AutoencoderKL, SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

logger = logging.get_logger(__name__)
T5_DIM = 4096

def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def retrieve_timesteps(scheduler, num_inference_steps=None, device=None, sigmas=None, **kwargs):
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


class StableDiffusion3Img2ImgPipelineWithBasePrompt(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    """
    Self-contained img2img pipeline for SD3.5 with base_prompt dual-conditioning.

    Encodes the location prompt through all three text encoders (CLIP-L, CLIP-G, T5)
    for scene/texture conditioning, and optionally re-encodes base_prompt through
    CLIP-L + CLIP-G only to produce object-anchored pooled embeddings.

    The full denoising loop runs against SD3.5's MMDiT transformer directly,
    with no delegation to any parent pipeline's __call__.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = [
        "tokenizer", "tokenizer_2", "tokenizer_3",
        "text_encoder", "text_encoder_2", "text_encoder_3",
    ]
    _callback_tensor_inputs = [
        "latents", "prompt_embeds", "negative_prompt_embeds",
        "pooled_prompt_embeds", "negative_pooled_prompt_embeds",
    ]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,       # CLIP-L
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,    # CLIP-G
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,                 # T5-XXL
        tokenizer_3: T5TokenizerFast,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = self.tokenizer.model_max_length if tokenizer is not None else 77
        self.default_sample_size = self.transformer.config.sample_size

    # ------------------------------------------------------------------ #
    #  Encoding helpers                                                    #
    # ------------------------------------------------------------------ #

    def _encode_clip(self, prompt, tokenizer, text_encoder, device,
                     num_images_per_prompt, clip_skip=None):
        """Encode prompt through one CLIP encoder. Returns (token_embeds, pooled_embeds)."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        output = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        pooled = output[0]
        token_embeds = output.hidden_states[-2] if clip_skip is None else output.hidden_states[-(clip_skip + 2)]

        bs, seq, dim = token_embeds.shape
        token_embeds = token_embeds.repeat(1, num_images_per_prompt, 1)
        token_embeds = token_embeds.view(bs * num_images_per_prompt, seq, dim)
        pooled = pooled.repeat(num_images_per_prompt, 1)
        return token_embeds, pooled

    def _encode_t5(self, prompt, device, num_images_per_prompt):
        """Encode prompt through T5-XXL. Returns token_embeds."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_3.model_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        embeds = self.text_encoder_3(text_inputs.input_ids.to(device))[0]
        bs, seq, dim = embeds.shape
        embeds = embeds.repeat(1, num_images_per_prompt, 1)
        embeds = embeds.view(bs * num_images_per_prompt, seq, dim)
        return embeds

    def _assemble_prompt_embeds(self, clip_l_embeds, clip_g_embeds, t5_embeds):
        """
        Assemble final prompt_embeds matching SD3 diffusers convention:
        - Pad CLIP-L (768) and CLIP-G (1280) token embeds each to T5_DIM (4096)
        - Concatenate along sequence dim: [clip_l_padded | clip_g_padded | t5_tokens]
        Result: (B, clip_seq + clip_seq + t5_seq, T5_DIM)
        """
        # Pad each CLIP output to T5_DIM along hidden dim
        clip_l_padded = F.pad(clip_l_embeds, (0, T5_DIM - clip_l_embeds.shape[-1]))  # (B, 77, 4096)
        clip_g_padded = F.pad(clip_g_embeds, (0, T5_DIM - clip_g_embeds.shape[-1]))  # (B, 77, 4096)
        # Concatenate along sequence dimension
        return torch.cat([clip_l_padded, clip_g_padded, t5_embeds], dim=1)            # (B, 77+77+t5_seq, 4096)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        base_prompt: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        clip_skip: Optional[int] = None,
    ):
        """
        Encode prompt with optional base_prompt for object-class pooled conditioning.

        prompt      → CLIP-L + CLIP-G token embeds + T5 token embeds  (scene/texture)
        base_prompt → CLIP-L + CLIP-G pooled embeds only              (object anchor)

        If base_prompt is None, pooled embeds come from the location prompt (standard SD3).
        """
        device = device or self._execution_device

        # --- Location prompt: all three encoders ---
        clip_l_embeds, clip_l_pooled = self._encode_clip(
            prompt, self.tokenizer, self.text_encoder,
            device, num_images_per_prompt, clip_skip,
        )
        clip_g_embeds, clip_g_pooled = self._encode_clip(
            prompt, self.tokenizer_2, self.text_encoder_2,
            device, num_images_per_prompt, clip_skip,
        )
        t5_embeds = self._encode_t5(prompt, device, num_images_per_prompt)

        prompt_embeds = self._assemble_prompt_embeds(clip_l_embeds, clip_g_embeds, t5_embeds)

        # --- Pooled embeddings ---
        if base_prompt is None:
            # Standard: pooled from location prompt CLIP encoders
            pooled_prompt_embeds = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)
        else:
            # base_prompt path: encode object label through CLIP only for pooled
            _, base_clip_l_pooled = self._encode_clip(
                base_prompt, self.tokenizer, self.text_encoder,
                device, num_images_per_prompt, clip_skip,
            )
            _, base_clip_g_pooled = self._encode_clip(
                base_prompt, self.tokenizer_2, self.text_encoder_2,
                device, num_images_per_prompt, clip_skip,
            )
            pooled_prompt_embeds = torch.cat([base_clip_l_pooled, base_clip_g_pooled], dim=-1)

            # --- Embedding interpolation (from original pipeline TODO, uncomment to experiment) ---
            # location_pooled = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)
            # alpha = 0.5  # 0 = full location, 1 = full base_prompt anchor
            # pooled_prompt_embeds = alpha * pooled_prompt_embeds + (1 - alpha) * location_pooled

        # --- Negative prompt ---
        if do_classifier_free_guidance:
            neg = negative_prompt or ""
            neg_clip_l_embeds, neg_clip_l_pooled = self._encode_clip(
                neg, self.tokenizer, self.text_encoder,
                device, num_images_per_prompt, clip_skip,
            )
            neg_clip_g_embeds, neg_clip_g_pooled = self._encode_clip(
                neg, self.tokenizer_2, self.text_encoder_2,
                device, num_images_per_prompt, clip_skip,
            )
            neg_t5_embeds = self._encode_t5(neg, device, num_images_per_prompt)

            negative_prompt_embeds = self._assemble_prompt_embeds(
                neg_clip_l_embeds, neg_clip_g_embeds, neg_t5_embeds
            )
            negative_pooled_prompt_embeds = torch.cat([neg_clip_l_pooled, neg_clip_g_pooled], dim=-1)
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    # ------------------------------------------------------------------ #
    #  Latent helpers                                                      #
    # ------------------------------------------------------------------ #

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt,
                        dtype, device, generator=None):
        image = image.to(device=device, dtype=dtype)
        effective_batch = batch_size * num_images_per_prompt

        if isinstance(generator, list) and len(generator) != effective_batch:
            raise ValueError(
                f"Generator list length {len(generator)} does not match "
                f"effective batch size {effective_batch}."
            )

        if isinstance(generator, list):
            init_latents = torch.cat(
                [retrieve_latents(self.vae.encode(image[i:i+1]), generator=generator[i])
                 for i in range(effective_batch)],
                dim=0,
            )
        else:
            init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        init_latents = init_latents.to(dtype=dtype)

        if effective_batch > init_latents.shape[0]:
            if effective_batch % init_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate image of batch {init_latents.shape[0]} to {effective_batch}."
                )
            init_latents = torch.cat(
                [init_latents] * (effective_batch // init_latents.shape[0]), dim=0
            )

        noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.scale_noise(init_latents, timestep[0].unsqueeze(0), noise)
        return latents

    def get_timesteps(self, num_inference_steps, strength, device):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        return timesteps, num_inference_steps - t_start

    # ------------------------------------------------------------------ #
    #  Main call — full denoising loop, no delegation to parent           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: PIL.Image.Image,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_skip: Optional[int] = None,
        sigmas: Optional[List[float]] = None,
        # --- our addition ---
        base_prompt: Optional[str] = None,
        **kwargs,
    ):
        device = self._execution_device
        do_cfg = guidance_scale > 1.0

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size

        # 1. Encode prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt[0] if batch_size == 1 else prompt,
            base_prompt=base_prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=(
                negative_prompt[0] if negative_prompt and len(negative_prompt) == 1
                else negative_prompt
            ),
            clip_skip=clip_skip,
        )

        # 2. Preprocess image
        image = self.image_processor.preprocess(image)

        # 3. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1]

        # 4. Prepare latents from the noised input image
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 5. Denoising loop against SD3.5 MMDiT transformer
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # Expand latents for CFG
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                timestep = t.expand(latent_model_input.shape[0])

                # Assemble embeddings for CFG
                if do_cfg:
                    embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    pooled_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                else:
                    embeds_input = prompt_embeds
                    pooled_input = pooled_prompt_embeds

                # MMDiT forward pass
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=embeds_input,
                    pooled_projections=pooled_input,
                    return_dict=False,
                )[0]

                # Classifier-free guidance
                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Flow matching scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {
                        k: locals()[k]
                        for k in callback_on_step_end_tensor_inputs
                        if k in locals()
                    }
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 6. Decode latents
        if output_type == "latent":
            image_out = latents
        else:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            image_out = self.vae.decode(latents, return_dict=False)[0]
            image_out = self.image_processor.postprocess(image_out, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)
        return StableDiffusion3PipelineOutput(images=image_out)