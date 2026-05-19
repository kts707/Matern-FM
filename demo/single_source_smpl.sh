
CONFIG=configs/moyo_fixed_source.yaml
OUT_DIR=demo/results/single_source_smpl
CKPT=ckpts/smpl_single_source.ckpt
SEED=2026
NUM_SAMPLES=5

python test/test_single_source.py \
        --config $CONFIG \
        --checkpoint $CKPT \
        --source_mesh demo/meshes/smpl/smpl_2.5k.obj \
        --output_dir ${OUT_DIR}/smpl_2.5k \
        --num_samples $NUM_SAMPLES \
        --repeat 1 \
        --seed $SEED \
        --render

python test/test_single_source.py \
        --config $CONFIG \
        --checkpoint $CKPT \
        --source_mesh demo/meshes/smpl/smpl_18k.obj \
        --output_dir ${OUT_DIR}/smpl_18k \
        --num_samples $NUM_SAMPLES \
        --repeat 1 \
        --seed $SEED \
        --render

python test/test_single_source.py \
        --config $CONFIG \
        --checkpoint $CKPT \
        --source_mesh demo/meshes/smpl/smpl_75k.obj \
        --output_dir ${OUT_DIR}/smpl_75k \
        --num_samples 1 \
        --repeat $NUM_SAMPLES \
        --seed $SEED \
        --render

python test/test_single_source.py \
        --config $CONFIG \
        --checkpoint $CKPT \
        --source_mesh demo/meshes/smpl/smpl_mixed.obj \
        --output_dir ${OUT_DIR}/smpl_mixed_res \
        --num_samples 1 \
        --repeat $NUM_SAMPLES \
        --seed $SEED \
        --render