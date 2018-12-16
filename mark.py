import cv2
import os


def display(url):
    gif = cv2.VideoCapture(url)
    print(url+':')
    gif.read()
    ret, frame = gif.read()
    cv2.imshow('0', frame)
    cv2.waitKey(1)
    gif.release()


def label(url, value):
    with open(url, 'w') as f:
        f.write(value)


def work_range(rdir):
    gif_len = len(os.listdir(rdir + 'gifs'))
    value_len = len(os.listdir(rdir + 'labels'))
    return range(value_len+1, gif_len+1)

def main():
    for i in work_range('./trains/'):
        gif_url = './trains/gifs/%06d.gif' % i
        label_url = './trains/labels/%06d.txt' % i
        display(gif_url)
        label(label_url, input())

    for i in work_range('./tests/'):
        gif_url = './trains/gifs/%06d.gif' % i
        label_url = url = './trains/labels/%06d.txt' % i
        display(gif_url)
        label(label_url, input())

if __name__ == '__main__':
    main()
