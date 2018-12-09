import requests
import os


def count(test=False):
    if test:
        url = './tests/gifs'
    else:
        url = './trains/gifs'
    gifs = os.listdir(url)
    n = len(gifs)
    return n


def download(num, test=False):
    if test:
        url = './tests/gifs/%06d.gif' % num
    else:
        url = './trains/gifs/%06d.gif' % num
    print('download' + url)
    response = requests.get('https://sais.jlu.edu.cn/auth.php')
    with open(url, 'wb') as f:
        f.write(response.content)


if __name__ == '__main__':
    tests_num = count(test=True)
    for i in range(100 - tests_num):
        tests_num += 1
        download(tests_num, test=True)
    trains_num = count()
    for i in range(1000 - trains_num):
        trains_num += 1
        download(trains_num)
