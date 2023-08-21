# git checkout origin/AAAI

rsync -avP "$@" --exclude   ".git" --exclude "wandb" --exclude "logs"  \
-e "ssh -p 2234" ./ parzival@140.118.127.153:~/Desktop/Datasets/OD/dfd-clip/

rsync -avP "$@"  \
-e "ssh -p 2234" ../dfd-clip/misc parzival@140.118.127.153:~/Desktop/Datasets/OD/dfd-clip/

rsync -avP "$@"  \
-e "ssh -p 2234" ../dfd-clip/datasets parzival@140.118.127.153:~/Desktop/Datasets/OD/dfd-clip/