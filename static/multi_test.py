import cv2
import numpy as np
from PIL import Image
import os
import shutil


class FaceCropper(object):
    CASCADE_PATH = "haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, filename):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if (faces is None):
            print('Failed to detect face')
            return 0
        # facecnt = len(faces)
        # print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]

        tmp_res=[]
        save_path = f"./uploads/{os.path.splitext(filename)[0]}"

        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            os.makedirs(save_path)

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (361, 361))
            i += 1
            tmp_res.append(lastimg)
            cv2.imwrite(f"{save_path}/src_{i}.jpg", lastimg)
        cv2.imwrite(f"./uploads/result.png", np.concatenate(tmp_res, axis=1) )
        return 0


FC = FaceCropper()
filename = '[2022-08-30-15:39]913dbb0406734382a36054f050e51b27_123.jpeg'
img_path = './uploads/[2022-08-30-15:39]913dbb0406734382a36054f050e51b27_123.jpeg'
FC.generate(img_path, os.path.splitext(filename)[0])

