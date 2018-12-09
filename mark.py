import cv2
import os


def display(num, test=False):
    if test:
        url = './tests/gifs/%06d.gif' % num
    else:
        url = './trains/gifs/%06d.gif' % num
    gif = cv2.VideoCapture(url)
    gif.read()
    ret, frame = gif.read()
    cv2.imshow('0', frame)
    cv2.waitKey(1)


def label(num, value, test=True):
    if test:
        url = './tests/labels/%06d.txt' % num
    else:
        url = './trains/labels/%06d.txt' % num
    with open(url, 'w') as f:
        f.write(value)


def work(test=False):
    if test:
        rdir = './tests/'
    else:
        rdir = './trains/'
    gif_len = len(os.listdir(rdir + 'gifs'))
    value_len = len(os.listdir(rdir + 'labels'))
    return range(value_len+1, gif_len+1)


if __name__ == '__main__':
    TEST = True
    for i in work(TEST):
        display(i, TEST)
        label(i, input('%06d.txt:' % i), TEST)
    TEST = False
    for i in work(TEST):
        display(i, TEST)
        label(i, input('%06d.txt:' % i), TEST)
