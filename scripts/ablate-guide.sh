python -m main --cfg configs/train/base-all.yaml --notes "baseline";\
python -m main --cfg configs/train/guide/skin+lips+nose.yaml --notes "baseline(3Q)";\
python -m main --cfg configs/train/guide/skin+lips.yaml --notes "baseline(2Q)";\
python -m main --cfg configs/train/guide/skin.yaml --notes "baseline(1Q)";