import cv2
import matplotlib.pyplot as plt
import numpy as np


def nearest_inter(img, dim_out):
    src_h, src_w, src_channel = img.shape
    dst_h, dst_w = dim_out
    scale_x, scale_y = dst_w/src_w, dst_h/src_h
    new_img = np.zeros((dst_h, dst_w, src_channel), np.uint8)
    for x in range(dst_w):
        for y in range(dst_h):
            # round / int/ math.floor
            new_img[y][x] = img[round(y/scale_y), round(x/scale_x)]
    return new_img


def gray_img_self(img):
    src_h, src_w, src_channel = img.shape
    new_img = np.zeros((src_h, src_w, 1), np.uint8)
    for y in range(src_h):
        for x in range(src_w):
            new_img[y][x] = np.mean([img[y, x, k] for k in range(3)])
    return new_img


def gray_img_api_cv2(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gray_img_api_skimage(img):
    from skimage.color import rgb2gray
    return rgb2gray(img)


def binary_show_0_1(img):
    # 0-1 之间的数组, 不能直接用cv2展现
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')


def biliear_inter(img, dim_out):
    src_h, src_w, chans = img.shape
    dst_h, dst_w = dim_out
    scale_x, scale_y = src_w/dst_w, src_h/dst_h
    dst_img = np.zeros((dst_h, dst_w, chans), np.uint8)
    for chan in range(chans):
        for x in range(dst_w):
            for y in range(dst_h):
                src_x, src_y = x*scale_x, y*scale_y
                src_x, src_y = (x+0.5)*scale_x-0.5, (y+0.5)*scale_y-0.5
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1, src_w-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1, src_h-1)
                tmp_0 = (src_x1-src_x)*img[src_y0][src_x0][chan] + \
                    (src_x-src_x0)*img[src_y0][src_x1][chan]
                tmp_1 = (src_x1-src_x)*img[src_y1][src_x0][chan] + \
                    (src_x-src_x0)*img[src_y1][src_x1][chan]
                dst_img[y, x, chan] = int(
                    (src_y1-src_y)*tmp_0 + (src_y-src_y0)*tmp_1)
    return dst_img


def histogram_gray_cv2(img):
    '''
    calcHist—计算图像直方图
    函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
    images：图像矩阵，例如：[image]
    channels：通道数，例如：0
    mask：掩膜，一般为：Non
    histSize：直方图大小，一般等于灰度级数
    ranges：横轴范围
    '''
    import matplotlib.pyplot as plt
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist_img = cv2.calcHist(img, [0], None, [256], [0, 256])
    print(hist)
    plt.figure('hist cv2')
    plt.title('cv2 -- gray -- calcHist')
    plt.plot(hist, label='gray img')
    plt.plot(cv2.calcHist([img], [0], None, [256],
                          [0, 256]), color='r', label='0')
    plt.plot(cv2.calcHist([img], [1], None, [256],
                          [0, 256]), color='g', label='1')
    plt.plot(cv2.calcHist([img], [2], None, [256],
                          [0, 256]), color='b', label='2')
    plt.plot(cv2.calcHist([cv2.split(img)[0]], [0],
                          None, [256], [0, 256]), color='y', label='y')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    plt.xlim([0, 256])
    plt.legend()
    plt.show()


def histogram_ravel(img):
    import matplotlib.pyplot as plt
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray_img.shape)
    print(gray_img.ravel().shape)
    plt.hist(gray_img.ravel(), 256)
    plt.show()


def histogram_equal_gray(img):
    '''
    equalizeHist—直方图均衡化
    函数原型： equalizeHist(src, dst=None)
    src：图像矩阵(单通道图像)
    dst：默认即可
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure()
    plt.title('equalize histogram')
    img_gray_equal_hist = cv2.equalizeHist(img_gray)
    print(img_gray.shape, img_gray_equal_hist.shape)
    plt.plot(cv2.calcHist([img_gray], [0], None,
                          [256], [0, 255]), label='gray')
    plt.plot(cv2.calcHist([img_gray_equal_hist], [0], None, [
             256], [0, 256]), label='gray-equal_hist')
    plt.hist(img_gray_equal_hist.ravel(), 256, label='ravel')

    plt.legend()
    plt.show()
    cv2.imshow('histogram equalize gray',
               np.hstack([img_gray, img_gray_equal_hist]))


def histogram_equal_color(img):
    chans = cv2.split(img)
    chans_eH = [cv2.equalizeHist(i) for i in chans]
    img_eh = cv2.merge(chans_eH)
    cv2.imshow('color histogram equalize', np.hstack([img, img_eh]))
    plt.figure()

    for index, chan in enumerate(chans):
        chan_ret = cv2.equalizeHist(chan)
        plt.plot(cv2.calcHist([chan], [0], None, [256],
                              [0, 255]), label='{}'.format(index))
        # plt.plot(cv2.calcHist([chan_ret],[0],None,[256],[0,255]))

    plt.plot(cv2.calcHist([img], [0], None, [256], [0, 255]), label='00')
    plt.plot(cv2.calcHist([img], [1], None, [256], [0, 255]), label='11')
    plt.plot(cv2.calcHist([img], [2], None, [256], [0, 255]), label='22')
    plt.legend()
    plt.show()


def sobel_filter(img):
    '''
    Sobel函数求完导数后会有负值，还有会大于255的值。
    而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
    因此要使用16位有符号的数据类型，即cv2.CV_16S。
    '''
    print(cv2.CV_32S, cv2.CV_16S, cv2.CV_8S)
    x = cv2.Sobel(img, cv2.CV_8S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_8S, 0, 1)
    '''
    在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
    否则将无法显示图像，而只是一副灰色的窗口。
    dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
    其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
    '''
    print(x)
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)
    cv2.imshow('abs_x', abs_x)
    cv2.imshow('abs_y', abs_y)
    '''
    由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
    。其函数原型为：
    dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
    其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
    gamma是加到最后结果上的一个值。
    '''
    sobel_img = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    cv2.imshow('sobel_img', sobel_img)


if __name__ == '__main__':
    img = cv2.imread('lenna.png', 3)
    print(img.shape)
    cv2.imshow('init_image', img)
    # # 灰度图
    # gray_img_self = gray_img_self(img)
    # cv2.imshow('gray-self img', gray_img_self)

    # gray_img_cv2 = gray_img_api_cv2(img)
    # cv2.imshow('gray-cv2 img', gray_img_cv2)

    # gray_img_skimage = gray_img_api_skimage(img)
    # cv2.imshow('gray-skimage img', gray_img_skimage)
    # print(gray_img_skimage.shape)
    # # 二值化
    # gray_img_skimage_binary1 = np.where(gray_img_skimage >= 0.5, 255, 0).astype(np.uint8)
    # gray_img_skimage_binary2 = np.where(gray_img_skimage >= 0.5, 1, 0)
    # cv2.imshow('image binary 0-1', gray_img_skimage_binary1)
    # binary_show_0_1(gray_img_skimage_binary2)

    # # 最近邻上采样
    # new_img = nearest_inter(img, (1000, 800))
    # print(new_img.shape)
    # cv2.imshow('nearest img', new_img)

    # # 双线性上采样
    # biliear_img = biliear_inter(img, (1024, 1024))
    # cv2.imshow('biliear img', biliear_img)

    # 计算直方图
    # histogram_gray_cv2(img)
    # histogram_ravel(img)
    # 直方图均衡化
    # histogram_equal_gray(img)
    # histogram_equal_color(img)

    # sobel 滤波
    sobel_filter(img)

    cv2.waitKey()
    cv2.destroyAllWindows()
