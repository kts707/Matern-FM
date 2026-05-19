CONFIG=configs/moyo_arbitrary_source.yaml
OUT_DIR=demo/results
SEED=2026

python test/test_arbitrary_source.py \
        --config $CONFIG \
        --checkpoint ckpts/smpl_arbitrary_source.ckpt \
        --source_dir demo/meshes/humanoids \
        --output_dir ${OUT_DIR}/arbitrary_source_humanoids \
        --repeat 5 \
        --seed $SEED \
        --render