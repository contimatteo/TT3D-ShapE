###
### Unibo CV-LAB Server run script 
###

exit 0

CUDA_VISIBLE_DEVICES=3 python3 tt3d_generate.py \
  --prompt-file ./data/prompts.txt \
  --out-path /media/data2/mconti/TT3D/models/ShapE/outputs/
