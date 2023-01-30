The preprocess code is borrowed from [RealForensics](https://github.com/ahaliassos/RealForensics)
for fair comparison. The only difference is our code splits the processed video into 1-sec clips
for more efficient training. We also use [Accelerate](https://huggingface.co/docs/accelerate/index)
for multi-gpu preprocessing.

To begin with, first we get the landmark of each video
```sh
accelerate launch get_landmark.py \
--data_dir [path to faceforensics dataset root]
```

Next, we crop and align the face
```sh
accelerate launch extract_single_aligned_face.py \
--data_dir [path to faceforensics dataset root] \
--save_dir ffpp-cropped-face
```

Lastly, don't forget to put the [split jsons](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits) under the dataset root