import cv2
import numpy as np


class MotionVector:
    def __init__(self,start,end):
        self.start = tuple(start)
        self.end = tuple(end)

    def range(self):
        return np.sqrt(np.square(np.subtract(self.end,self.start)).sum())


def rgb2gray(img):
    """
    :param img: input 3 channel to be turned into gray
    :return: gray image
    """
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def gray2bgr(img):
    return cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

def blur_img(img):
    """
    :param img: input image to be blurred
    :return: blurred image
    """
    return cv2.GaussianBlur(img, (3, 3), 0)


def block_swap(block_pairs,anchor):

    new_image = np.zeros_like(anchor)
    block_size = block_pairs[0][0].block_size

    for block_pair in block_pairs:
        anchor_block,target_block = block_pair
        target_center = target_block.center
        data = anchor_block.data

        new_image[
        (target_center[0]-block_size//2):(target_center[0]+block_size//2),
        (target_center[1]-block_size//2):(target_center[1]+block_size//2)
        ] = data

    return new_image

def get_motion_vectors(block_pairs):
    vectors = []

    for block_pair in block_pairs:
        anchor = block_pair[0]
        target = block_pair[1]

        vectors.append(MotionVector(start=anchor.center,end=target.center))

    return vectors

def draw_motion_vector(image,vector,color=(0,0,255),thickness=1):
    try:
        if vector.range==0.0:
            image = cv2.circle(image,center=vector.start, radius=0, color=color,thickness=thickness)
        else:
            image = cv2.arrowedLine(image, pt1=vector.start, pt2=vector.end, color=color, thickness=thickness)
        return image
    except Exception:
        print("Make sure you input list of vectors!")
        return None

def draw_motion_vectors(image,vectors,color=(0,0,255),thickness=1):
    if isinstance(vectors,list):
        for vector in vectors:
            image = draw_motion_vector(image=image,vector=vector,color=color,thickness=thickness)
        return image
    else:
        return None