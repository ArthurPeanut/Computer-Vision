import sys
import cv2
import random as rand

import numpy as np

import GraphOperator as go

def generate_image(ufset, width, height):
    random_color = lambda: (int(rand.random() * 255), int(rand.random() * 255), int(rand.random() * 255))
    color = [random_color() for i in range(width * height)]

    save_img = np.zeros((height, width, 3), np.uint8)

    for y in range(height):
        for x in range(width):
            color_idx = ufset.find(y * width + x)
            save_img[y, x] = color[color_idx]

    return save_img

def main():
    sigma = float(sys.argv[1])
    k = float(sys.argv[2])
    min_size = float(sys.argv[3])

    img = cv2.imread(sys.argv[4])
    float_img = np.asarray(img, dtype=float)

    gaussian_img = cv2.GaussianBlur(float_img, (5, 5), sigma)
    b, g, r = cv2.split(gaussian_img)
    smooth_img = (r, g, b)

    height, width, channel = img.shape
    graph = go.build_graph(smooth_img, width, height)

    weight = lambda edge: edge[2]
    sorted_graph = sorted(graph, key=weight)

    ufset = go.segment_graph(sorted_graph, width * height, k)
    ufset = go.remove_small_component(ufset, sorted_graph, min_size)

    """
    img_gt = cv2.imread(sys.argv[5], 0)
    gt_img = np.asarray(img_gt, dtype=int)


    gt_pixel_cnt = 0
    total_pixel_cnt = 0
    s = set()
    cnt = 0
    for y in range(height):
        for x in range(width):
            idx = ufset.find(y * width + x)
            prev_size = len(s)
            s.add(idx)
            # 如果是没求过的集：
            if prev_size != len(s):
                total_pix = 0
                gt_pix = 0
                for yy in range(height):
                    for xx in range(width):
                        if ufset.find(yy * width + xx) == idx:
                            total_pix += 1
                            if gt_img[yy, xx] == 255:
                                gt_pix += 1
                if (gt_pix / total_pix) > 0.5:
                    gt_pixel_cnt += gt_pix
                    total_pixel_cnt += total_pix
                cnt += 1

            if cnt == 50:
                break
        if cnt == 50:
            break

    print("IOU=%f" % gt_pix/total_pix)
    """
    save_img = generate_image(ufset, width, height)
    cv2.imwrite(sys.argv[5], save_img)

if __name__ == '__main__':
    main()