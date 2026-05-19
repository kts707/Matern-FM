CONFIG=$1
OUT_DIR=$2

SEED=2026

python test/test_single_source.py \
        --config $CONFIG \
        --source_mesh demo/meshes/stanford_bunny/bunny_10k.obj \
        --output_dir ${OUT_DIR}/bunny_10k \
        --num_samples 5 \
        --repeat 1 \
        --seed $SEED \
        --render

python test/test_single_source.py \
        --config $CONFIG \
        --source_mesh demo/meshes/stanford_bunny/bunny_70k.obj \
        --output_dir ${OUT_DIR}/bunny_70k \
        --num_samples 5 \
        --repeat 1 \
        --seed $SEED \
        --render

python test/test_single_source.py \
        --config $CONFIG \
        --source_mesh demo/meshes/stanford_bunny/bunny_mixed.obj \
        --output_dir ${OUT_DIR}/bunny_mixed_res \
        --num_samples 1 \
        --repeat 5 \
        --seed $SEED \
        --render 