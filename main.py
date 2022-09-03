from math import log10, sqrt
import cv2
from skimage.metrics import structural_similarity
import imutils
import numpy as np


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def SSIM(original, compressed):
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

    (ssim, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    # print("SSIM: {}".format(ssim))
    return ssim


def main():
    # foler = open("original")  # image0000.jpeg, image0001.png
    # folderze2 = open("constructed/")  # image0000.png, image0001.png
    original = cv2.imread("original.jpg")
    compressed = cv2.imread("compressed.png", 1)

    psnr = PSNR(original, compressed)
    ssim = SSIM(original, compressed)

    print(f"PSNR value is {psnr} dB and SSIM: {format(ssim)}")


if __name__ == "__main__":
    main()
