from typing import Tuple, Iterator, Any
from pathlib import Path

import torch

from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_mesh
from utils import Utils

###

device = Utils.Cuda.init()

###


def _load_models() -> Any:
    xm = load_model('transmitter', device=device)
    return xm


def _load_latents() -> Iterator[Tuple[str, torch.Tensor]]:
    source_path = Path("outputs", "latents")

    print("")
    for prompt_path in source_path.rglob("*"):
        if prompt_path.is_dir():
            filename = "latents.pt"
            filepath = prompt_path.joinpath(filename)
            print(prompt_path.name)
            assert filepath.exists() and filepath.is_file()
            prompt = prompt_path.name.replace("_", " ")
            yield prompt, torch.load(filepath)
    print("")


def _convert_latents_to_objs(xm_model: Any) -> None:
    assert xm_model is not None

    latents_iter = _load_latents()
    out_path = Path("outputs", "meshes")

    for prompt, latents in latents_iter:
        for idx, latent in enumerate(latents):
            tri_mesh = decode_latent_mesh(xm_model, latent).tri_mesh()

            prompt_dirname = prompt.replace(" ", "_")
            file_basepath = out_path.joinpath(prompt_dirname)
            file_basepath.mkdir(parents=True, exist_ok=True)

            ply_filepath = file_basepath.joinpath(f"mesh_{idx}.ply")
            with open(ply_filepath, 'wb') as f:
                tri_mesh.write_ply(f)

            obj_filepath = file_basepath.joinpath(f"mesh_{idx}.obj")
            with open(obj_filepath, 'w', encoding="utf-8") as f:
                tri_mesh.write_obj(f)


###


def main() -> None:
    xm_model = _load_models()
    _convert_latents_to_objs(xm_model=xm_model)


###

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--prompt', type=str, required=True)
    # args = parser.parse_args()

    #

    main()
