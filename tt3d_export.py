### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, Any, List, Iterator
from pathlib import Path

import argparse
import torch

from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_mesh
from utils import Utils

###

T_Prompt = Tuple[str, Path]  ### pylint: disable=invalid-name
# T_Prompts = List[T_Prompt]  ### pylint: disable=invalid-name
T_Prompts = Iterator[T_Prompt]  ### pylint: disable=invalid-name

device = Utils.Cuda.init()

###


def _load_models() -> Any:
    xm = load_model('transmitter', device=device)
    return xm


def _load_prompts_from_source_path(source_rootpath: Path) -> T_Prompts:
    assert isinstance(source_rootpath, Path)
    assert source_rootpath.exists()
    assert source_rootpath.is_dir()

    experiment_path = Utils.Storage.build_experiment_path(out_rootpath=source_rootpath)

    for prompt_path in experiment_path.iterdir():
        if prompt_path.is_dir():
            prompt_dirname = prompt_path.name
            yield (prompt_dirname, prompt_path)


def _convert_latents_to_objs(
    prompt: str,
    source_rootpath: Path,
    xm_model: Any,
) -> None:
    assert xm_model is not None

    source_prompt_latents_filepath = Utils.Storage.build_prompt_latents_filepath(
        out_rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=True,
    )

    latents = torch.load(source_prompt_latents_filepath)

    #

    assert len(latents) == 1
    latent = latents[0]

    # for idx, latent in enumerate(latents):
    mesh = decode_latent_mesh(xm_model, latent).tri_mesh()

    out_ply_filepath = Utils.Storage.build_prompt_mesh_filepath(
        out_rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=False,
        # idx=idx,
        extension="ply",
    )
    out_ply_filepath.parent.mkdir(parents=True, exist_ok=True)
    out_obj_filepath = Utils.Storage.build_prompt_mesh_filepath(
        out_rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=False,
        # idx=idx,
        extension="obj",
    )
    out_obj_filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(out_ply_filepath, 'wb+') as f:
        mesh.write_ply(f)
    with open(out_obj_filepath, 'w+', encoding="utf-8") as f:
        mesh.write_obj(f)


###


def main(source_rootpath: Path,) -> None:
    assert isinstance(source_rootpath, Path)
    assert source_rootpath.exists()
    assert source_rootpath.is_dir()

    xm_model = _load_models()
    prompts = _load_prompts_from_source_path(source_rootpath=source_rootpath)

    print("")
    for prompt_enc, _ in prompts:
        prompt = Utils.Prompt.decode(prompt_enc)

        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        print("")
        print(prompt)

        _convert_latents_to_objs(
            prompt=prompt,
            source_rootpath=source_rootpath,
            xm_model=xm_model,
        )
        print("")
    print("")


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source-path', type=Path, required=True)
    # parser.add_argument('--out-path', type=Path, required=True)
    ### TODO: add option to skip existing objs.
    # parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(source_rootpath=args.source_path)
