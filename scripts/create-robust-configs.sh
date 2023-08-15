TYPES="CS CC BW GNC GB JPEG VC"
LEVELS="1 2 3 4 5"
for T in $TYPES; 
do
    for L in $LEVELS; 
        do
        cat << EOT > "configs/inference/robustness/$T($L).yaml" 
data:
  eval:
    - category: Deepfake
      types:
        - REAL
        - NT
        - DF
        - FS
        - F2F
      name: FFPP
      pack: true
      compressions:
        - c23
      root_dir: ./datasets/robustness/$T/$L/
EOT
    done
done