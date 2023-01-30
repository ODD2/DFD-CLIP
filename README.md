### Environment preparation
Use conda to build the environment
```sh
conda env create -f environment.yml
```
We use [Accelerate](https://huggingface.co/docs/accelerate/index) to run multi-gpu training. Please setup the environment according to your device by
```sh
accelerate config
```

### Data preparation
Please follow the instruction in [preprocessing](preprocessing)

### Training
An example training on FaceForensics cross-manipulation benchmark on **raw** videos
```sh
./scripts/cross-manipulation-train.sh
```
We randomly sample an 1-sec clip for every videos for each epoch during training.

### Testing
To test the performance of the example above
```sh
./scripts/cross-manipulation-test.sh
```
We test all 1-sec clips in each video and average the prediction

#### Result
We run the example above on 10 Nvidia RTX A5000

|          | Deepfakes | Face2Face | FaceSwap | NeuralTextures |
|----------|-----------|-----------|----------|----------------|
| Accuracy | 97.8      | 96.0      | 90.4     | 94.4           |
| AUROC    | 100.      | 99.8      | 97.9     | 97.5           |
