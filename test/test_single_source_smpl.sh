CONFIG=$1
OUT_DIR=$2

SEED=2026

python test/test_single_source.py \
        --config $CONFIG \
        --source_mesh demo/meshes/smpl/smpl_2.5k.obj \
        --output_dir ${OUT_DIR}/smpl_2.5k \
        --num_samples 5 \
        --repeat 1 \
        --seed $SEED \
        --render

python test/test_single_source.py \
        --config $CONFIG \
        --source_mesh demo/meshes/smpl/smpl_18k.obj \
        --output_dir ${OUT_DIR}/smpl_18k \
        --num_samples 5 \
        --repeat 1 \
        --seed $SEED \
        --render

python test/test_single_source.py \
        --config $CONFIG \
        --source_mesh demo/meshes/smpl/smpl_75k.obj \
        --output_dir ${OUT_DIR}/smpl_75k \
        --num_samples 1 \
        --repeat 5 \
        --seed $SEED \
        --render

python test/test_single_source.py \
        --config $CONFIG \
        --source_mesh demo/meshes/smpl/smpl_mixed.obj \
        --output_dir ${OUT_DIR}/smpl_mixed_res \
        --num_samples 1 \
        --repeat 5 \
        --seed $SEED \
        --render