
CONFIG=configs/fish.yaml
OUT_DIR=demo/results/single_source_fish
CKPT=ckpts/fish.ckpt
SEED=2026
NUM_SAMPLES=5

python test/test_single_source.py \
        --config $CONFIG \
        --checkpoint $CKPT \
        --source_mesh demo/meshes/fish/fish_7k.obj \
        --output_dir ${OUT_DIR}/fish_7k \
        --num_samples $NUM_SAMPLES \
        --repeat 1 \
        --seed $SEED \
        --render 