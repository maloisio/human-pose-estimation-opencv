# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2
import cv2 as cv
import numpy as np
import argparse
from scipy.spatial import distance as dist

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.04, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"],["RShoulder", "LShoulder"] ]

inWidth = 368
inHeight = 368

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

#cap = cv.VideoCapture(args.input if args.input else 0)0
#cap = cv.imread("image.jpg")

#imgWidth = cap.shape[1]
#imgHeight = cap.shape[0]
#imgChanel = cap.shape[2]


#img2 = np.zeros([imgWidth, imgHeight, imgChanel], np.uint8)  # criacao imagem preta

while cv.waitKey(1) < 0:
    frame = cv.imread("cbum3.jpg")
#while cv.waitKey(1) < 0:
 #   hasFrame, frame = cap.read()
  #  if not hasFrame:
   #     cv.waitKey()
    #    break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

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
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            #print(points[idTo][0])
            distEuc = dist.euclidean((points[idTo][0], points[idTo][1]), (points[idFrom][0], points[idFrom][1]))
            if distEuc <=300:
                print(distEuc)
                #print(points[idFrom])
                #print(points[idTo])
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.putText(frame, str(int(distEuc)), (int(points[idTo][0]), int(points[idTo][1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                if (points[4] and points[7]) is not None:
                    if points[4][1] and points[7][1] < points[0][1]:
                        print("esse " + str(points[4][1])+", "+ str(points[7][1]))
                        print("esse " + str(points[0][1]))
                        cv.putText(frame, "Assalto", (0,70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)

                #if (points[9] and points[12]) is not None:
                  #  print("Nao e none")
                   # if points[12][0] < points[9][0]:#quando peca cruzado
                    #    #cv.line(frame, points[idFrom], points[idTo], (0, 255, 255), 3)
                     #   cv.line(frame, points[9], points[8], (0, 255, 255), 3)
                      #  cv.line(frame, points[12], points[11], (0, 255, 255), 3)
                #else:
                 #   print("nONE")




            #cv.line(img2, points[idFrom], points[idTo], (0, 255, 0), 3)
            #cv.ellipse(img2, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            #cv.ellipse(img2, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.putText(frame, str(cv2.CAP_PROP_FPS), (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('OpenPose using OpenCV', frame)
    cv.imshow('OpenPose using OpenCV2', heatMap)
    #cv.imshow('OpenPose using OpenCV2', img2)