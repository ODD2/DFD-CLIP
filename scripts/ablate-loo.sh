python -m main --cfg configs/train/loo/DF.yaml --aux_file "configs/inference/loo/DF.yaml" --notes "baseline(DF)";\
python -m main --cfg configs/train/loo/FS.yaml --aux_file "configs/inference/loo/FS.yaml"  --notes "baseline(FS)";\
python -m main --cfg configs/train/loo/F2F.yaml --aux_file "configs/inference/loo/F2F.yaml"  --notes "baseline(F2F)";\
python -m main --cfg configs/train/loo/NT.yaml --aux_file "configs/inference/loo/NT.yaml"  --notes "baseline(NT)";