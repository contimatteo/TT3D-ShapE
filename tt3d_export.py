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


# def _load_latents() -> Iterator[Tuple[str, torch.Tensor]]:
#     source_path = Path("outputs", "latents")
#     print("")
#     for prompt_path in source_path.rglob("*"):
#         if prompt_path.is_dir():
#             filename = "latents.pt"
#             filepath = prompt_path.joinpath(filename)
#             print(prompt_path.name)
#             assert filepath.exists() and filepath.is_file()
#             prompt = prompt_path.name.replace("_", " ")
#             yield prompt, torch.load(filepath)
#     print("")


def _load_prompts_from_source_path(source_path: Path) -> T_Prompts:
    assert isinstance(source_path, Path)
    assert source_path.exists()
    assert source_path.is_dir()

    # prompts: T_Prompts = []
    # for prompt_path in source_path.rglob("*"):
    for prompt_path in source_path.iterdir():
        if prompt_path.is_dir():
            prompt_dirname = prompt_path.name
            # prompts.append((prompt_dirname, prompt_path))
            yield (prompt_dirname, prompt_path)
    # return prompts


# def _convert_latents_to_objs(
#     out_rootpath: Path,
#     xm_model: Any,
#     prompts: T_Prompts,
# ) -> None:
#     assert isinstance(out_rootpath, Path)
#     assert out_rootpath.exists()
#     assert out_rootpath.is_dir()
#     assert xm_model is not None

#     print(">")
#     for prompt_dirname, prompt_path in prompts:
#         assert isinstance(prompt_dirname, str)
#         assert isinstance(prompt_path, Path)

#         latents_path = prompt_path.joinpath("ckpts", "latents.pt")
#         print(">")
#         print(">", latents_path)
#         print(">")
#         assert latents_path.exists()
#         assert latents_path.is_file()
#         latents = torch.load(latents_path)

#         out_path = out_rootpath.joinpath(prompt_dirname, "meshes")
#         out_path.mkdir(exist_ok=True, parents=True)

#         for idx, latent in enumerate(latents):
#             tri_mesh = decode_latent_mesh(xm_model, latent).tri_mesh()

#             ply_filepath = out_path.joinpath(f"mesh_{idx}.ply")
#             with open(ply_filepath, 'wb') as f:
#                 tri_mesh.write_ply(f)

#             obj_filepath = out_path.joinpath(f"mesh_{idx}.obj")
#             with open(obj_filepath, 'w', encoding="utf-8") as f:
#                 tri_mesh.write_obj(f)
#     print(">")


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

    for idx, latent in enumerate(latents):
        mesh = decode_latent_mesh(xm_model, latent).tri_mesh()

        out_ply_filepath = Utils.Storage.build_prompt_mesh_filepath(
            out_rootpath=source_rootpath,
            prompt=prompt,
            assert_exists=False,
            idx=idx,
            extension="ply",
        )
        out_ply_filepath.parent.mkdir(parents=True, exist_ok=True)
        out_obj_filepath = Utils.Storage.build_prompt_mesh_filepath(
            out_rootpath=source_rootpath,
            prompt=prompt,
            assert_exists=False,
            idx=idx,
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
    prompts = _load_prompts_from_source_path(source_path=source_rootpath)

    # _convert_latents_to_objs(
    #     out_rootpath=out_path,
    #     xm_model=xm_model,
    #     prompts=prompts,
    # )

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
