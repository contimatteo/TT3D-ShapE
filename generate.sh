###

exit 0


ROOT_DIR="/media/data2/mconti/TT3D"
PROMPT_DIR="${ROOT_DIR}/prompts"
OUT_DIR="${ROOT_DIR}/outputs/test_t3bench_n1"

GPU=0
PROMPT_FILE="${PROMPT_DIR}/test.t3bench.n1.txt"


###


CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/ShapE/"
# --skip-existing
