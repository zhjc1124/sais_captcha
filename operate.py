import cv2
import numpy as np
import os
VALID_CHARS = '23456789abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPRSTUVWXYZ'


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_gradient(x):
    return sigmoid(x)*(1-sigmoid(x))


def get_frames(url):
    gif = cv2.VideoCapture(url)
    gif.read()
    frames = []
    for i in range(15):
        ret, frame = gif.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        else:
            break
    return frames


def cut_frame(frame):
    char_vecs = None
    for i in range(4):
        left = 18*i
        char_vec = frame[8:24, left:left+8].flatten()
        if char_vecs is None:
            char_vecs = char_vec
        else:
            char_vecs = np.vstack((char_vecs, char_vec))
    return char_vecs


def gif2vec(url):
    frames = get_frames(url)
    vecs = None
    for frame in frames:
        vec = cut_frame(frame)
        if vecs is None:
            vecs = vec
        else:
            vecs = np.vstack((vecs, vec))
    return vecs/255


def label2vec(url):
    with open(url, 'r') as f:
        label = f.read()
    return np.array(list(map(char2vec, label)) * 15)


def char2vec(c):
    vec = np.array(list(VALID_CHARS)) == c
    vec = vec.astype(np.int32)
    return vec


def vec2char(y):
    chars_arr = np.array(list(VALID_CHARS))

    labels = np.array([])
    for vec in y:
        labels = np.append(labels, chars_arr[np.max(vec) == vec])
    return labels


def get_arrs(test=False):
    if test:
        rdir = './tests'
    else:
        rdir = './trains'
    num = len(os.listdir(rdir + '/labels/'))
    print('valid labeled gif num: %d' % num)
    x_arr = None
    for i in range(1, num+1):
        x_vec = gif2vec(rdir + '/gifs/%06d.gif' % i)
        if x_arr is None:
            x_arr = x_vec
        else:
            x_arr = np.vstack((x_arr, x_vec))

    y_arr = None
    for i in range(1, num+1):
        y_vec = label2vec(rdir + '/labels/%06d.txt' % i)
        if y_arr is None:
            y_arr = y_vec
        else:
            y_arr = np.vstack((y_arr, y_vec))

    return x_arr, y_arr


def main():
    x, y = get_arrs()
    print('x_train shape:', x.shape)
    print('y_train shape:', y.shape)
    np.save('x.npy', x)
    np.save('y.npy', y)
    x_test, y_test = get_arrs(test=True)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)


if __name__ == '__main__':
    main()
    # y = np.load('y.npy')
    # print(vec2char(y))




