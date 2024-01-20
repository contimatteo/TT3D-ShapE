import env_setup

################################################################################

from typing import Dict, Tuple, Any, Optional

import argparse
import torch

from pathlib import Path
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import GaussianDiffusion
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model
from shap_e.models.download import load_config
from shap_e.util.notebooks import create_pan_cameras
from shap_e.util.notebooks import decode_latent_images
from shap_e.util.notebooks import gif_widget

from utils import Utils

###

T_Model = Dict[str, torch.Tensor]

device = Utils.Cuda.device()

print("")
# print(f'available: {torch.cuda.is_available()}')
# print(f'available devices: {torch.cuda.device_count()}')
# torch.cuda.set_device(1)
# print(f'current device: { torch.cuda.current_device()}')
# print("")
# for i in range(torch.cuda.device_count()):
#    print(torch.cuda.get_device_properties(i).name)
print("Is cuda available?", torch.cuda.is_available())
print("Is cuDNN version:", torch.backends.cudnn.version())
print("cuDNN enabled? ", torch.backends.cudnn.enabled)
print("Device count?", torch.cuda.device_count())
print("Current device?", torch.cuda.current_device())
print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))
print("")

print("")
print("")
raise Exception("stop")

###


def _args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--prompt', type=str, required=True)


def _load_models(
        xm: bool) -> Tuple[T_Model, GaussianDiffusion, Optional[T_Model]]:
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    xm = xm if load_model('transmitter', device=device) else None
    return model, diffusion, xm


def _sample_latents(
    prompt: str,
    model: T_Model,
    diffusion: GaussianDiffusion,
    batch_size: int = 4,
    guidance_scale: float = 15.0,
) -> Any:
    assert isinstance(prompt, str)
    assert len(prompt) > 0

    ### TODO: map all params to config file ...
    return sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )


def _store_latents(out_path: Path, latents: Any) -> None:
    assert isinstance(out_path, Path)
    assert out_path.exists()
    assert out_path.is_dir()
    # assert isinstance(latents, ?)


###


def main(prompt: str):
    model, diffusion, _ = _load_models(xm=False)
    latents = _sample_latents(prompt=prompt, model=model, diffusion=diffusion)

    print("")
    print("")
    print(type(latents))
    print("")
    print("")

    return

    out_prompt_dir_name = prompt.strip().replace(" ", "_")
    out_path = Path(".")
    out_path.joinpath("outputs", out_prompt_dir_name)
    out_path.mkdir(exist_ok=True, parents=True)
    _store_latents(latents=latents, out_path=out_path)


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    _args(parser=parser)
    args = parser.parse_args()

    #

    main(prompt=args.prompt)
