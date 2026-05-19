CONFIG=$1
OUT_DIR=$2

SEED=2026

python test/test_single_source.py \
        --config $CONFIG \
        --source_mesh demo/meshes/fish/fish_7k.obj \
        --output_dir ${OUT_DIR}/fish_7k \
        --num_samples 5 \
        --repeat 1 \
        --seed $SEED \
        --render 