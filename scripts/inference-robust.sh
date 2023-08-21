TYPES="CS CC BW GNC GB JPEG VC"
LEVELS="1 2 3 4 5"
for T in $TYPES; 
do
    for L in $LEVELS; 
        do
        python -m inference "$1" "configs/inference/robustness/$T($L).yaml" --notes "Robust:$T($L)"
    done
done