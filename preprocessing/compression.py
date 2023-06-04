import os
import cv2
from tqdm import tqdm
from glob import glob


files = glob("/storage/HCI/Sessions/*/*C_Section*.avi")
print(f"Working on {len(files)} files.")
print(f"The first file is {files[0]}.")
input("Press to Continue...")
for s in tqdm(sorted(files)):
    d = s.replace("Sessions", "FrontCam-C23")
    d_folder = "/".join(d.split('/')[:-1])
    if os.path.exists(d) and os.path.exists(s):
        cap = cv2.VideoCapture(s)
        s_fps = round(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        cap = cv2.VideoCapture(d)
        d_fps = round(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if(s_fps == d_fps):
            continue
        else:
            os.system(f"rm -rf {d_folder}")

    os.makedirs(d_folder, exist_ok=True)
    os.system(f'/usr/bin/ffmpeg -hide_banner -loglevel error -y -i "{s}" -crf 23 -c:v libx264 -r 61 "{d}"')
