import cv2
import imageio


def denoise(img):
    pass

if __name__ == "__main__":
    gif = imageio.mimread('two/auth.gif')
    for index, image in enumerate(gif):
        cv2.imwrite('two/%d.jpg' % index, image)
