# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import math
from collections import deque

import cv2
import cv2 as cv
import numpy as np
import argparse
from scipy.spatial import distance as dist
import threading

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
                help="max buffer size")
args1 = vars(ap.parse_args())

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. pSkip to capture frames from camera')
parser.add_argument('--thr', default=0.16, type=float, help='Threshold value for pose parts heat map')
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

inWidth = 366
inHeight = 366
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture("serving2.mp4")

pointsToAngle = []
pointsList = []

paintedPoints = deque(maxlen=300)
savedPaintedPoints = []
frameArray = []
interestPoint = 0


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
    try:
        angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    except ZeroDivisionError:
        angR = 0
    angD = round(math.degrees(angR))
    if angD < 0:
        angD = 180 + angD
    cv.putText(frame2, str(angD), pt1, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
    print(angD)


def getLBodyAngle(pt1, pt2, pt3):
    m1 = gradiant(pt1, pt2)
    m2 = gradiant(pt1, pt3)
    try:
        angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    except ZeroDivisionError:
        angR = 0
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


def funcButton():
    pass


while True:
    hasFrame, frame = cap.read()
    key = cv.waitKey(1)

    #---------------loop video---------------
    if not hasFrame:
         cv.destroyAllWindows()
         cap = cv.VideoCapture("serving2.mp4")
         hasFrame, frame = cap.read()

    #frame = cv.rotate(frame, cv.ROTATE_180);
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = rescale_frame(frame, percent=50)

    frame = frame[50:900, 150:600]
    #frame = cv.resize(frame, [255, 255], interpolation=cv.INTER_BITS)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame2 = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)  # criacao imagem preta

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 0.5, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False))
    out = net.forward()
    out = out[:, :19, :, : ]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

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

            distEuc = dist.euclidean((points[idTo][0], points[idTo][1]), (points[idFrom][0], points[idFrom][1]))
            if distEuc <= 700:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 1)
                cv.line(frame2, points[idFrom], points[idTo], (0, 255, 0), 1)
                cv.imshow('Frame2', frame2)
                cv.ellipse(frame, points[idFrom], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame2, points[idFrom], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame2, points[idTo], (2, 2), 0, 0, 360, (0, 0, 255), cv.FILLED)

            # ----------------------DESENHA LINHA MOTION TRACKING--------------------

            #mao direita
            if key == ord("1"):
                paintedPoints.clear()
                interestPoint = 4
            # mao equerda
            if key == ord("2"):
                paintedPoints.clear()
                interestPoint = 7
            # joelho direito
            if key == ord("3"):
                paintedPoints.clear()
                interestPoint = 9
            # joelho esquerdo
            if key == ord("4"):
                paintedPoints.clear()
                interestPoint = 12
            # quadril direito
            if key == ord("5"):
                paintedPoints.clear()
                interestPoint = 8
            # quadril esquerdo
            if key == ord("6"):
                paintedPoints.clear()
                interestPoint = 11
            #pe direito
            if key == ord("7"):
                paintedPoints.clear()
                interestPoint = 10
            #pe esquerdo
            if key == ord("8"):
                paintedPoints.clear()
                interestPoint = 13

            if interestPoint is not 0:
                paintedPoints.appendleft(points[interestPoint])
                savedPaintedPoints.append(points[interestPoint])

            for i in np.arange(1, len(paintedPoints)):
                if paintedPoints[i - 1] is None or paintedPoints[i] is None:
                    continue
                #thickness = int(np.sqrt(args1["buffer"] / float(i + 1)) * 2.5)
                cv.line(frame, paintedPoints[i - 1], paintedPoints[i], (0, 0, 255), 2)
                cv.line(frame2, paintedPoints[i - 1], paintedPoints[i], (0, 0, 255), 2)

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

    #frame = cv.resize(frame, [400, 400], interpolation=cv.INTER_BITS)
    frameArray.append(frame2)
    cv.imshow('Frame', frame)
    cv.imshow('Frame2', frame2)
    heatMap = cv.resize(heatMap, [400, 400], interpolation=cv.INTER_BITS)
    cv.imshow('Pontos', heatMap)


    if key == ord('q'):
        break
    if key == ord('p'):
        cv.waitKey(-1)  # wait until any key is pressed

    #------------------MANIPULANDO VIDEO/FRAMES-----------------
    if key == ord('j'):
        cv.destroyWindow('Frame')
        cv.destroyWindow('Frame2')
        cv.destroyWindow('Pontos')
        frame3 = np.zeros((255, 255, 3), np.uint8)
        cv.imshow('Frame3', frameArray[0])
        counter = 0
        i = 0
        while key != ord('q'):

            #-------------START/PAUSE------------------
            if key == ord('p'):
                while True:
                    if i > 0:
                        for i in np.arange(i, len(frameArray)):
                            cv.imshow('Frame3', frameArray[i])
                            key = cv.waitKey(50)
                            if key == ord('r'):
                                i = 0
                                break
                            if key == ord('p'):
                                break
                    else:
                        for i in np.arange(1, len(frameArray)):
                            cv.imshow('Frame3', frameArray[i])
                            key = cv.waitKey(50)
                            if key == ord('r'):
                                i = 0
                                break
                            if key == ord('p'):
                                break
                    break

            #-----------AVANÇAR FRAME-------------
            if key == ord('l'):
                if i < len(frameArray) -1:
                    i = 1 + i
                    cv.imshow('Frame3', frameArray[i])

            #----------VOLTAR FRAME--------------
            if key == ord('k'):
                if i > 0:
                    i = i - 1
                    cv.imshow('Frame3', frameArray[i])

            #----------RESTART------------------
            if key == ord('r'):
                i = 0
                cv.imshow('Frame3', frameArray[i])

            key = cv.waitKey(1)
            if key == ord('q'):
                break
        cv.destroyAllWindows()

cap.release()
cv.destroyAllWindows()
