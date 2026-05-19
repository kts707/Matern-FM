CONFIG=$1
OUT_DIR=$2

SEED=2026

python test/test_arbitrary_source.py \
        --config $CONFIG \
        --source_dir demo/meshes/humanoids \
        --output_dir ${OUT_DIR}/humanoids_results \
        --repeat 5 \
        --seed $SEED \
        --render 