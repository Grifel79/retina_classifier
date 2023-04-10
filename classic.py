import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

def left_right(path, show = False):
    im = cv2.imread(path)
    im = cv2.resize(im, (750, 500))

    if show:
        cv2.imshow('image', im)
        cv2.waitKey(0)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if show:
        cv2.imshow('image', im_gray)
        cv2.waitKey(0)

    _, im_retina = cv2.threshold(im_gray,5,255,cv2.THRESH_BINARY)
    if show:
        cv2.imshow('image', im_retina)
        cv2.waitKey(0)
    M_r = cv2.moments(im_retina)
    if M_r["m00"] == 0:
        return -1
    # calculate x,y coordinate of center
    cX_r = int(M_r["m10"] / M_r["m00"])
    cY_r = int(M_r["m01"] / M_r["m00"])

    #print(cX_r, cY_r)

    _, im_nerve = cv2.threshold(im_gray,130,255,cv2.THRESH_BINARY)
    if show:
        cv2.imshow('image', im_nerve)
        cv2.waitKey(0)
    # calculate moments of binary image
    M_n = cv2.moments(im_nerve)
    if M_n["m00"] == 0:
        return -1
    # calculate x,y coordinate of center
    cX_n = int(M_n["m10"] / M_n["m00"])
    cY_n = int(M_n["m01"] / M_n["m00"])

    #print(cX_n, cY_n)

    if cX_n >= cX_r:
        return("right")
    elif cX_n < cX_r:
        return("left")

retina_train = "train"

val_ds = tf.keras.utils.image_dataset_from_directory(
  retina_train,
  validation_split=0.2,
  subset="validation",
  seed=123,                 # same seed as in training!!!
  image_size=(1000, 1500),
  batch_size = 1)

file_paths = val_ds.file_paths

for path in file_paths:
    if path!=None:
        # i can get the correct label just from the image's path
        if "left" in path:
            label = "left"
        elif "right" in path:
            label = "right"

        # here i print all misclassified val images:
        if label != left_right(path):
            print(path)
            # left_right(path, show=True)

