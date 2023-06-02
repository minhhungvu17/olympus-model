import os
import cv2
import platform
import numpy as np
import logging
import json


def get_rotate_crop_image(img, points):
    # Use Green's theory to judge clockwise or counterclockwise
    # author: biyanhua
    d = 0.0
    for index in range(-1, 3):
        d += (
            -0.5
            * (points[index + 1][1] + points[index][1])
            * (points[index + 1][0] - points[index][0])
        )
    if d < 0:  # counterclockwise
        tmp = np.array(points)
        points[1], points[3] = tmp[3], tmp[1]

    try:
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    except Exception as e:
        print(e)


def getImglabelidx(filePath):
    if platform.system() == "Windows":
        spliter = "\\"
    else:
        spliter = "/"
    filepathsplit = filePath.split(spliter)[-2:]
    return filepathsplit[0] + "/" + filepathsplit[1]


def saveRecResult(PPlabelpath):
    rec_gt_dir = os.path.dirname(PPlabelpath) + "/rec_gt.txt"
    crop_img_dir = os.path.dirname(PPlabelpath) + "/crop_img/"
    dir_path = os.path.dirname(PPlabelpath)
    # label_gt_dir = os.path.dirname(PPlabelpath) + '/Label.txt'
    ques_img = []

    if not os.path.exists(crop_img_dir):
        os.makedirs(crop_img_dir)

    with open(PPlabelpath, "r", encoding="utf-8") as fr:
        with open(rec_gt_dir, "w", encoding="utf-8") as f:
            lines = fr.readlines()
            for line in lines:
                path, gt = line.split("\t")
                print(path)
                gt = json.loads(gt)
                try:
                    name = os.path.basename(path)
                    img = cv2.imread(os.path.join(dir_path, name))

                    for i, label in enumerate(gt):
                        img_crop = get_rotate_crop_image(
                            img, np.array(label["points"], np.float32)
                        )
                        img_name = (
                            os.path.splitext(os.path.basename(path))[0]
                            + "_crop_"
                            + str(i)
                            + ".jpg"
                        )
                        cv2.imwrite(crop_img_dir + img_name, img_crop)
                        f.write("crop_img/" + img_name + "\t")
                        f.write(label["transcription"] + "\n")
                except Exception as e:
                    ques_img.append(path)
                    print("Can not read image ", e)
                
        if ques_img:
            logging.info(
                "The Following images can not be saved, please check the image path and labels"
            )


if __name__ == "__main__":
    # path = "/Users/vietlq4/fpthealthcare/datasets/PXL_20221225_081748904~2/Label.txt"
    path = '/Users/vietlq4/Downloads/AI_MayHA_Frames_annotation'
    folders = ['ANDUA1020', 'B2basic', 'ANDUA767','B3adv', 'Facarehand','FacareTD3128','FORA']
    for folder in os.listdir(path):
        if folder in folders:
            for root, dirs, files in os.walk(os.path.join(path, folder)):
                for file in files:
                    if 'Label.txt' in os.path.join(root, file):
                        saveRecResult(os.path.join(root, file))
        
    # saveRecResult(path)
