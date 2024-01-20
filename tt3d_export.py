from typing import Tuple, Dict, Iterator

import argparse
import torch

from pathlib import Path
from shap_e.models.download import load_model
# from shap_e.util.notebooks import create_pan_cameras
from shap_e.util.notebooks import decode_latent_images
# from shap_e.util.notebooks import gif_widget

from utils import Utils

###

T_Model = Dict[str, torch.Tensor]
T_Latents = Dict[str, torch.Tensor]

device = Utils.Cuda.init()

###


def _load_models() -> T_Model:
    xm = load_model('transmitter', device=device)
    return xm


def _load_latents(path: Path) -> Iterator[T_Latents]:
    assert isinstance(path, Path)
    assert path.exists()
    assert path.is_dir()

    print("")
    for prompt_path in path.rglob("*"):
        if prompt_path.is_dir():
            print(prompt_path.name)
    print("")

    raise Exception("TODO: implement this function!")


def _convert_latents_to_objs(
    xm_model: T_Model,
    latents: T_Latents,
    out_path=Path,
) -> None:
    assert isinstance(xm_model, T_Model)
    assert isinstance(latents, T_Latents)
    assert isinstance(out_path, Path)
    ### Example of saving the latents as meshes.
    from shap_e.util.notebooks import decode_latent_mesh

    # for i, latent in enumerate(latents):
    #     t = decode_latent_mesh(xm, latent).tri_mesh()
    #     with open(f'example_mesh_{i}.ply', 'wb') as f:
    #         t.write_ply(f)
    #     with open(f'example_mesh_{i}.obj', 'w') as f:
    #         t.write_obj(f)


###


def main(source_latents_path: Path, out_obj_path: Path) -> None:
    xm_model = _load_models()
    latents = _load_latents(path=source_latents_path)
    _convert_latents_to_objs(
        xm_model=xm_model,
        latents=latents,
        out_path=out_obj_path,
    )


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--source-latents-path', type=Path, required=True)
    parser.add_argument('--out-obj-path', type=Path, required=True)

    args = parser.parse_args()

    #

    main(
        source_latents_path=args.source_latents_path,
        out_obj_path=args.out_obj_path,
    )
