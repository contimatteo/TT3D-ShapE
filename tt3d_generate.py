### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, Any

import argparse
import torch

from pathlib import Path
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import GaussianDiffusion
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model
from shap_e.models.download import load_config

from utils import Utils

###

device = Utils.Cuda.init()

###


def _load_models() -> Tuple[Any, GaussianDiffusion]:
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    return model, diffusion


def _generate_latents(
    prompt: str,
    out_rootpath: Path,
    model: Any,
    diffusion: GaussianDiffusion,
    skip_existing: bool,
    batch_size: int = 4,
    guidance_scale: float = 15.0,
    karras_steps: int = 64,
) -> None:
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert isinstance(batch_size, int)
    assert 1 <= batch_size <= 1000  ### avoid naive mistakes ...
    assert isinstance(guidance_scale, float)
    assert isinstance(karras_steps, int)
    assert 1 <= karras_steps <= 1000  ### avoid naive mistakes ...

    out_prompt_latents_filepath = Utils.Storage.build_prompt_latents_filepath(
        out_rootpath=out_rootpath,
        prompt=prompt,
        assert_exists=False,
    )

    if skip_existing and out_prompt_latents_filepath.exists():
        print("latents already exists -> ", out_prompt_latents_filepath)
        return

    out_prompt_latents_filepath.parent.mkdir(parents=True, exist_ok=True)

    #

    model_kwargs = dict(texts=[prompt] * batch_size)

    ### TODO: map all params to config file ...
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=model_kwargs,
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=karras_steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    assert isinstance(latents, torch.Tensor)

    torch.save(latents, out_prompt_latents_filepath)


# def _store_latents(
#     out_rootpath: Path,
#     prompt: str,
#     latents: torch.Tensor,
# ) -> None:
#     assert isinstance(out_rootpath, Path)
#     assert out_rootpath.exists()
#     assert out_rootpath.is_dir()
#     assert isinstance(prompt, str)
#     assert isinstance(latents, torch.Tensor)

#     prompt_dirname = Utils.Prompt.encode(prompt)
#     out_path = out_rootpath.joinpath(prompt_dirname, "ckpts")
#     out_path.mkdir(exist_ok=True, parents=True)

#     filename = "latents.pt"
#     out_path = out_path.joinpath(filename)

#     torch.save(latents, out_path)

###


def main(
    prompt_filepath: Path,
    out_rootpath: Path,
    skip_existing: bool,
    batch_size: int,
    karras_steps: int,
):
    assert isinstance(prompt_filepath, Path)
    assert isinstance(out_rootpath, Path)
    assert isinstance(batch_size, int)
    assert isinstance(skip_existing, bool)

    if out_rootpath.exists():
        assert out_rootpath.is_dir()
    else:
        out_rootpath.mkdir(parents=True)

    #

    model, diffusion = _load_models()
    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    print("")
    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        print("")
        print(prompt)

        _generate_latents(
            prompt=prompt,
            out_rootpath=out_rootpath,
            model=model,
            diffusion=diffusion,
            skip_existing=skip_existing,
            batch_size=batch_size,
            karras_steps=karras_steps,
        )
        print("")
    print("")


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--out-path', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--karras-steps', type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(
        prompt_filepath=args.prompt_file,
        out_rootpath=args.out_path,
        skip_existing=args.skip_existing,
        batch_size=args.batch_size,
        karras_steps=args.karras_steps,
    )
