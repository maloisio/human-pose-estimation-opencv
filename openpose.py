# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import math

import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cv2 as cv
import numpy as np
import argparse
from scipy.spatial import distance as dist

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["LShoulder", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["RShoulder", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["LShoulder", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["LHip", "RHip"]]

inWidth = 255
inHeight = 255
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture("mauro4.mp4")
pointsToAngle = []
pointsList = []


# cap = cv.imread("operator1.jpg")

# imgWidth = cap.shape[1]
# imgHeight = cap.shape[0]
# imgChanel = cap.shape[2]


# while cv.waitKey(1) < 0:
# frame = cv.imread("mauro.mp4")
def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_BITS)


def getRBodyAngle(pt1, pt2, pt3):
    m1 = gradiant(pt1, pt2)
    m2 = gradiant(pt1, pt3)
    angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(math.degrees(angR))
    if angD < 0:
        angD = 180 + angD
    cv.putText(frame2, str(angD), pt1, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
    print(angD)


def getLBodyAngle(pt1, pt2, pt3):
    m1 = gradiant(pt1, pt2)
    m2 = gradiant(pt1, pt3)
    angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(math.degrees(angR))
    if angD < 0:
        angD = 180 + angD
    cv.putText(frame2, str(angD), pt1, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv.LINE_AA)
    print(angD)


def getAngle(pointList):
    pt1, pt2, pt3 = pointList
    m1 = gradiant(pt1, pt2)
    m2 = gradiant(pt1, pt3)
    angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(math.degrees(angR))
    if angD <= 0:
        angD = 180 + angD
    print(angD)


def gradiant(pt1, pt2):
    try:
        (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    except ZeroDivisionError:
        return 1000
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def mousePoints(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(pointsList) >= 3:
            pointsList.clear()
        cv.circle(frame, (x, y), 5, (0, 0, 255), cv.FILLED)
        cv.imshow('Frame', frame)
        pointsList.append([x, y])
        print(pointsList)
        getAngle(pointsList)
def nothing():
    pass


while True:

    hasFrame, frame = cap.read()
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    frame = rescale_frame(frame, percent=50)
    frame = frame[50:700, 300:730]
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    rgb_planes = cv.split(frame)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 21)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        norm_img = cv.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

        frame = cv.merge(result_planes)
        result_norm = cv.merge(result_norm_planes)

    frame2 = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)  # criacao imagem preta

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 0.5, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False))
    out = net.forward()
    out = out[:7, :19, :-1, :-1]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert (len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this wayy.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            # print(points[idTo][0])
            distEuc = dist.euclidean((points[idTo][0], points[idTo][1]), (points[idFrom][0], points[idFrom][1]))
            if distEuc <= 700:
                # print(distEuc)
                # print(points[idFrom])
                # print(points[idTo])
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 1)
                cv.line(frame2, points[idFrom], points[idTo], (0, 255, 0), 1)
                cv.imshow('Frame2', frame2)
                # cv.putText(frameCrop, str(int(distEuc)), (int(points[idTo][0]), int(points[idTo][1])),
                # cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
                cv.ellipse(frame, points[idFrom], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame2, points[idFrom], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame2, points[idTo], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)

                # if (points[4] and points[7]) is not None:
                # if points[4][1] and points[7][1] < points[0][1]:
                # print("esse " + str(points[4][1])+", "+ str(points[7][1]))
                # print("esse " + str(points[0][1]))
                # cv.putText(frame, "Assalto", (0,70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)

                # if (points[9] and points[12]) is not None:
                #  print("Nao e none")
                # if points[12][0] < points[9][0]:#quando peca cruzado
                #    #cv.line(frame, points[idFrom], points[idTo], (0, 255, 255), 3)
                #   cv.line(frame, points[9], points[8], (0, 255, 255), 3)
                #  cv.line(frame, points[12], points[11], (0, 255, 255), 3)
                # else:
                #   print("nONE")

            # cv.line(img2, points[idFrom], points[idTo], (0, 255, 0), 3)
            # cv.ellipse(img2, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            # cv.ellipse(img2, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.putText(frame, str(cv.CAP_PROP_FPS), (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # ------------------------ANGLES-------------------

    # cotovelo direito
    if (points[3] and points[2] and points[4]) is not None:
        if points[3][0] > points[4][0]:
            getRBodyAngle(points[3], points[4], points[2])
        else:
            getRBodyAngle(points[3], points[2], points[4])

    # joelho direito
    if (points[9] and points[8] and points[10]) is not None:
        if points[9][0] > points[10][0]:
            getRBodyAngle(points[9], points[10], points[8])
        else:
            getRBodyAngle(points[9], points[8], points[10])

    # cotovelo esquerdo
    if (points[6] and points[7] and points[5]) is not None:
        if points[6][0] < points[7][0]:
            getLBodyAngle(points[6], points[5], points[7])
        else:
            getLBodyAngle(points[6], points[7], points[5])

    # joelho esquerdo
    if (points[12] and points[13] and points[11]) is not None:
        if points[12][0] < points[13][0]:
            getLBodyAngle(points[12], points[11], points[13])
        else:
            getLBodyAngle(points[12], points[13], points[11])

    if len(pointsList) == 3:
        getAngle(pointsList)

    cv.imshow('Frame', frame)
    cv.imshow('Frame2', frame2)
    #cv.imshow('shadows_out.png', result)
    #cv.imshow('shadows_out_norm.png', result_norm)

    cv.setMouseCallback("Frame", mousePoints)
    heatMap = cv.resize(heatMap, [400,400], interpolation=cv.INTER_BITS)
    cv.imshow('OpenPose using OpenCV2', heatMap)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv.waitKey(-1)  # wait until any key is pressed

cap.release()
cv.destroyAllWindows()
