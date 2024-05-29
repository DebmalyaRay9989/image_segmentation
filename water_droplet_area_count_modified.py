# %%
import cv2
import numpy as np
import glob
import csv
import os
import matplotlib.pyplot as plt
import argparse

# %%

#image = cv2.imread("water_droplets23.jpg")
#path = "water_droplets23.jpg"
#file1 = glob.glob(path)

# initialize the list of reference points and boolean indicating


def unsharp_mask(image, blur_size = (5,5), imgWeight = 1.5, gaussianWeight = -0.5):
        gaussian = cv2.GaussianBlur(image, (5,5), 0)
        return cv2.addWeighted(image, imgWeight, gaussian, gaussianWeight, 0)

# whether cropping is being performed or not
#refPt = []
#cropping = False

#image = unsharp_mask(image)

#mask = np.zeros(image.shape, dtype=np.uint8)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#med_blur = cv2.medianBlur(gray, ksize=3)
#_, thresh = cv2.threshold(med_blur, 190, 255, cv2.THRESH_BINARY)
#blending = cv2.addWeighted(gray, 0.5, thresh, 0.9, gamma=0)
#thresh = cv2.threshold(gray,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#edged = cv2.Canny(blending, 30, 200)


# %%
#cv2.__version__

# Import DictWriter class from CSV module
from csv import DictWriter

# %%
# Find contours and filter using contour area
#contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#center_moments= cv2.moments(contours[0])
#contours = contours[0:-1] # takes out contour bounding the whole image

#cv2.drawContours(image = edged,
#                 contours = contours,
#                 contourIdx = -1,
#                 color = (0, 0, 255),
#                 thickness = 5)


def mouse_call_back(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(0, len(contours)):
            r = cv2.pointPolygonTest(contours[i], (y, x), False)
            print(r)
            if r > 0:
                print("Selected contour ", i)


# %%
#total_area = 0
#l1 = []
#l2 = []
#smallest = sorted(contours, key=cv2.contourArea)[0]
#largest = sorted(contours, key=cv2.contourArea)[-1]


# %%
# Visualize result better
#result = cv2.bitwise_and(image, image, mask=thresh)
#result[thresh==0] = (255,255,255)


#cv2.namedWindow("res")
#cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#for idx, c in enumerate(contours):  # numbers the contours
#    x = int(sum(c[:,0,0]) / len(c))
#    y = int(sum(c[:,0,1]) / len(c))
#    #x,y,w,h = cv2.boundingRect(c)
#    area = cv2.contourArea(c)
#    if (area > 4.0):
#        l1.append(area)
#        image = cv2.drawContours(image, [c], 0, (222, 49, 99), 3)
#        cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 204), 2)
#    total_area += area
#cv2.imshow("res", image)

#print("*************************************************************************************************")
#print("*************************************************************************************************")
#print("The Image Name : ", file1)
#print("Number of Droplets found : " + str(len(l1)))
#print("The list of area of each water droplets : ", l1)
#print('Total area: {}'.format(total_area))
#print("**************************************************************************************************")
#print("**************************************************************************************************")
#file_exists = os.path.isfile('result3.csv')
#with open('result3.csv', 'a', newline='') as f_object:
#    field_names = ['NAME', 'DROPLET_COUNT_APPROX', 'DROPLETS_SIZE_LIST', 'TOTAL_AREA_COVERED']
#    writer = csv.DictWriter(f_object,  delimiter=',', lineterminator='\n',fieldnames=field_names)
#    if not file_exists:
#        writer.writeheader()
#    writer.writerow({'NAME': file1,
#                    'DROPLET_COUNT_APPROX': str(len(l1)),
#                    'DROPLETS_SIZE_LIST': l1,
#                    'TOTAL_AREA_COVERED': total_area})
#    cv2.waitKey()
#    f_object.close()


# %%

