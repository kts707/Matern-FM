
CONFIG=configs/stanford_bunny.yaml
OUT_DIR=demo/results/single_source_stanford_bunny
CKPT=ckpts/stanford_bunny.ckpt
SEED=2026
NUM_SAMPLES=5

python test/test_single_source.py \
        --config $CONFIG \
        --checkpoint $CKPT \
        --source_mesh demo/meshes/stanford_bunny/bunny_10k.obj \
        --output_dir ${OUT_DIR}/bunny_10k \
        --num_samples $NUM_SAMPLES \
        --repeat 1 \
        --seed $SEED \
        --render 

python test/test_single_source.py \
        --config $CONFIG \
        --checkpoint $CKPT \
        --source_mesh demo/meshes/stanford_bunny/bunny_mixed.obj \
        --output_dir ${OUT_DIR}/bunny_mixed \
        --num_samples $NUM_SAMPLES \
        --repeat 1 \
        --seed $SEED \
        --render

python test/test_single_source.py \
        --config $CONFIG \
        --checkpoint $CKPT \
        --source_mesh demo/meshes/stanford_bunny/bunny_70k.obj \
        --output_dir ${OUT_DIR}/bunny_70k \
        --num_samples 1 \
        --repeat $NUM_SAMPLES \
        --seed $SEED \
        --render