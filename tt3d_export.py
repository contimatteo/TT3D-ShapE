from typing import Tuple, Dict, Iterator, Any

import argparse
import torch

from pathlib import Path
from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_images
from shap_e.util.notebooks import decode_latent_mesh

from utils import Utils

###

T_Latents = Tuple[str, torch.Tensor]

device = Utils.Cuda.init()

###


def _load_models() -> Any:
    xm = load_model('transmitter', device=device)
    return xm


def _load_latents(path: Path) -> Iterator[T_Latents]:
    assert isinstance(path, Path)
    assert path.exists()
    assert path.is_dir()

    # prompts_latents_map: T_Latents = {}

    print("")
    for prompt_path in path.rglob("*"):
        if prompt_path.is_dir():
            print(prompt_path.name)
            filename = "latents.pt"
            filepath = prompt_path.joinpath(filename)
            assert filepath.exists() and filepath.is_file()
            prompt = prompt_path.name.replace("_", " ")
            # prompts_latents_map[prompt] = torch.load(filepath)
            yield prompt, torch.load(filepath)
    print("")


def _convert_latents_to_objs(
    xm_model: Any,
    source_path: Path,
    out_path=Path,
) -> None:
    assert xm_model is not None
    assert isinstance(source_path, Path)
    assert isinstance(out_path, Path)

    latents_iter = _load_latents(path=source_path)

    for idx, (prompt, latent) in enumerate(latents_iter):
        tri_mesh = decode_latent_mesh(xm_model, latent).tri_mesh()

        prompt_dirname = prompt.replace(" ", "_")
        file_basepath = out_path.joinpath(prompt_dirname, "meshes", idx)
        file_basepath.mkdir(parents=True, exist_ok=True)

        ply_filepath = file_basepath.joinpath("mesh.ply")
        with open(ply_filepath, 'wb') as f:
            tri_mesh.write_ply(f)

        obj_filepath = file_basepath.joinpath("mesh.obj")
        with open(obj_filepath, 'w', encoding="utf-8") as f:
            tri_mesh.write_obj(f)


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
