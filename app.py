
import os
from flask import Flask, render_template, request, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
import cv2
import pandas as pd
import seaborn as sns
import glob
import csv
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import base64
from PIL import Image
import base64
import io
import json
import math
# Import DictWriter class from CSV module
from csv import DictWriter
from water_droplet_area_count_modified import unsharp_mask
from water_droplet_area_count_modified import mouse_call_back
from skimage.measure import regionprops, regionprops_table
import pyclesperanto_prototype as cle

STATIC_FOLDER = os.path.join('static', 'result_photo')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = STATIC_FOLDER
img5 = os.path.join('static', 'Image')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index')
def show_index():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image_res.jpg')
    return render_template("index.html", user_image = full_filename)

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    
    # Get image
    if request.method == 'POST':
    
        image = request.files['image']
       ## file1 = glob.glob(image)
        file1 = str(image)
        #image = cv2.imread("water_droplets23.jpg")
        filestr = request.files['image'].read()
        file_bytes = np.fromstring(filestr, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        image = unsharp_mask(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        med_blur = cv2.medianBlur(gray, ksize=3)
        _, thresh = cv2.threshold(med_blur, 190, 255, cv2.THRESH_BINARY)
        blending = cv2.addWeighted(gray, 0.5, thresh, 0.9, gamma=0)
        thresh = cv2.threshold(gray,0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]        

        edged = cv2.Canny(blending, 30, 200)
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_moments= cv2.moments(contours[0])
        stats = regionprops(cle.pull(thresh).astype(int))
        regions = regionprops(cle.pull(thresh).astype(int))
        props = regionprops_table(
                cle.pull(thresh).astype(int),
                properties=('centroid', 'orientation', 'axis_major_length', 'axis_minor_length'),)

        
        #print(stats[0]['minor_axis_length'])
        #print(stats[0]['major_axis_length'])
        cv2.drawContours(image = thresh,
                 contours = contours,
                 contourIdx = -1,
                 color = (0, 0, 255),
                 thickness = 5)
                 
        # %%
        total_area = 0
        total_perimeter = 0
        equi_diameter = 0

        l1 = []
        l2 = []
        l3 = []
        smallest = sorted(contours, key=cv2.contourArea)[0]
        largest = sorted(contours, key=cv2.contourArea)[-1]
        
        
        # %%
        # Visualize result better
        result = cv2.bitwise_and(image, image, mask=thresh)
        result[thresh==0] = (255,255,255)
        
        # cv2.namedWindow("res")
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        
        for idx, c in enumerate(contours):  # numbers the contours
            x = int(sum(c[:,0,0]) / len(c))
            y = int(sum(c[:,0,1]) / len(c))
            #x,y,w,h = cv2.boundingRect(c)
            #####  ============== Contour Area ============= #####
            area = cv2.contourArea(c)
            #####  ============== Contour Perimeter ============= #####
            perimeter = cv2.arcLength(c,True)
            ##### ==============  Equivalent Diameter =========== ######
            ##### np.sqrt(4*area/np.pi) #########
            equi_diameter = np.sqrt(4*area/np.pi)
            if (area > 4.0):
                l1.append(area)
                l2.append(perimeter)
                l3.append(equi_diameter)
                image = cv2.drawContours(image, [c], 0, (222, 49, 99), 3)
                cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 204), 2)
            total_area += area
            total_perimeter += perimeter
   
        # cv2.imshow("res", image)
        path = 'static/result_photo/'
        cv2.imwrite(str(path) + "image_res.jpg", image) 
        
        print("*************************************************************************************************")
        print("*************************************************************************************************")
        print("The Image Name : ", file1)
        print("Number of Droplets found : " + str(len(l1)))
        print("The list of area of each water droplets : ", l1)
        print("The list of perimeter of each water droplets : ", l2)
        print("The list of equivalent diameter of each water droplets : ", l3)
        print('Total area: {}'.format(total_area))
        print('Total Perimeter: {}'.format(total_perimeter))
        props2 = pd.DataFrame(props)
       # print(props2)
        ext_centroid_1 = props2["centroid-0"].values.tolist()
        ext_centroid_2 = props2["centroid-1"].values.tolist()
        major_axes_1 = props2["axis_major_length"].values.tolist()
        minor_axes_1 = props2["axis_minor_length"].values.tolist()
        print("Major Axes: {}".format(major_axes_1))
        print("Minor Axes: {}".format(minor_axes_1))
        print("**************************************************************************************************")
        print("**************************************************************************************************")
        
        context = {
            'NAME': file1,
            'DROPLET_COUNT_APPROX': str(len(l1)),
            'DROPLETS_SIZE_LIST': l1,
            'DROPLETS_PERIMETER_LIST': l2,
            'DROPLETS_EQUIVALENT_DIAMETER': l3,
            'TOTAL_AREA_COVERED': total_area,
            'TOTAL_PERIMETER_COVERED': total_perimeter,
            'MAJOR_AXES': str(major_axes_1),
            'MINOR_AXES': str(minor_axes_1)
        }
               
        file_exists = os.path.isfile('result3.csv')
        
        #with open('result3.csv', 'a', newline='') as f_object:
        with open('result3.csv', 'a', newline='') as f_object:
            f_object.truncate()
            field_names = ['NAME', 'DROPLET_COUNT_APPROX', 'DROPLETS_SIZE_LIST', 'DROPLETS_PERIMETER_LIST', 'DROPLETS_EQUIVALENT_DIAMETER', 'TOTAL_AREA_COVERED', 'TOTAL_PERIMETER_COVERED', 'MAJOR_AXES', 'MINOR_AXES']
            writer = csv.DictWriter(f_object,  delimiter=',', lineterminator='\n',fieldnames=field_names)
            if not file_exists:
                writer.writeheader()
            writer.writerow({'NAME': file1,
                            'DROPLET_COUNT_APPROX': str(len(l1)),
                            'DROPLETS_SIZE_LIST': l1,
                            'DROPLETS_PERIMETER_LIST': l2,
                            'DROPLETS_EQUIVALENT_DIAMETER': l3,
                            'TOTAL_AREA_COVERED': total_area,
                            'TOTAL_PERIMETER_COVERED': total_perimeter,
                            'MAJOR_AXES': str(major_axes_1),
                            'MINOR_AXES': str(minor_axes_1)})
            cv2.waitKey()
            f_object.close()


        return render_template('index.html', NAME=file1, DROPLET_COUNT_APPROX=str(len(l1)), DROPLETS_SIZE_LIST=l1, DROPLETS_PERIMETER_LIST=l2, DROPLETS_EQUIVALENT_DIAMETER=l3, TOTAL_AREA_COVERED=total_area, MAJOR_AXIS=str(major_axes_1), MINOR_AXIS=str(minor_axes_1), TOTAL_PERIMETER_COVERED=total_perimeter, MAJOR_AXES=str(major_axes_1), MINOR_AXES=str(minor_axes_1), image=image)
 
    else:

        return render_template('index.html')


@app.route('/viewplot', methods=['GET', 'POST'])

def viewplot():
    
    if request.method == 'POST':
        
        df1 = pd.read_csv("result3.csv")
        print(df1.columns)
        df1 = df1.iloc[::-1]
        print(df1.shape)
        col_vals = df1["DROPLETS_SIZE_LIST"].values.tolist()
        col_vals = col_vals[0]
        print(col_vals)
        import ast
        list2 = ast.literal_eval(col_vals)
        print(type(list2))
        
        from matplotlib import rcParams

        # figure size in inches
        # rcParams['figure.figsize'] = 21.7,12.27
        output_dir = 'static/result_photo'
        #fig = ax.get_figure()
        fig = plt.figure(figsize = (10, 5))
        plt.bar(np.arange(len(list2)), list2, color ='maroon', width = 0.4)
        plt.axis('on')
        plt.xlabel("Length of the List")
        plt.ylabel("Entities within the List")
        plt.savefig('static/result_photo/plot.jpg')
        #fig.savefig('{}/plot.jpg'.format(output_dir))

    return render_template('viewplots.html')

@app.route('/viewplot2', methods=['GET', 'POST'])

def viewplot2():
    
    if request.method == 'POST':
        
        df2 = pd.read_csv("result3.csv")
        print(df2.columns)
        df2 = df2.iloc[::-1]
        print(df2.shape)
        col_vals = df2["DROPLETS_SIZE_LIST"].values.tolist()
        col_vals = col_vals[0]
        print(col_vals)
        import ast
        list2 = ast.literal_eval(col_vals)
        print(type(list2))
        
        from matplotlib import rcParams

        # figure size in inches
        # rcParams['figure.figsize'] = 21.7,12.27
        output_dir = 'static/result_photo'
        #fig = ax.get_figure()
        fig = plt.figure(figsize = (10, 5))
        plt.plot(np.arange(len(list2)), list2, color="#6c3376", linewidth=3)
        plt.axis('on')
        plt.xlabel("Length of the List")
        plt.ylabel("Entities within the List")
        plt.savefig('static/result_photo/plot2.jpg')
        #fig.savefig('{}/plot2.jpg'.format(output_dir))
        
    return render_template('viewplots2.html')

@app.errorhandler(Exception)  
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    # now you're handling non-HTTP exceptions only
    return render_template("500_generic.html", e=e), 500
#def handle_exception(e):
#    """Return JSON instead of HTML for HTTP errors."""
#    # start with the correct headers and status code from the error
#    response = e.get_response()
#    # replace the body with JSON
#    response.data = json.dumps({
#        "error": {
#            "code": e.code,
#            "name": e.name,
#            "description": e.description,
#        }
#    })
#    print(response.data) 
#    response.content_type = "application/json"
#    return response

if __name__ == "__main__":
    app.run(debug=True)
    
 

