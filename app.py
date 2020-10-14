import os
from flask import Flask,render_template,request
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
import random
import cv2
import csv
import matplotlib.image as mpimg
from skimage.feature import hog 
import joblib
import glob
import subprocess
from selenium import webdriver
import time
count=0


app=Flask(__name__,template_folder="templates",static_folder="images")
APP_ROOT=os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/signpage")
def signpage():
    return render_template("sign.html")

@app.route("/lanepage")
def lanepage():
    rm_mp4()
    return render_template("lane.html")

@app.route("/vehiclepage")
def vehiclepage():
    return render_template("vehicle.html")

def rm_jpg():
    for f in glob.glob("C:/Users/Cline/Desktop/TraFreak/images/results/*.jpg"):
        os.remove(f)
        
        
def rm_mp4():
    for f in glob.glob("C:/Users/Cline/Desktop/TraFreak/images/results/*.mp4"):
        os.remove(f) 


profile = webdriver.FirefoxProfile()
profile.set_preference("browser.cache.disk.enable", False)
profile.set_preference("browser.cache.memory.enable", False)
profile.set_preference("browser.cache.offline.enable", False)
profile.set_preference("network.http.use-cache", False) 
driver =webdriver.Firefox(profile)

    
def modified_model():    
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
model = modified_model()

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def signpredict(destination):
    reader = csv.reader(open('C:/Users/Cline/Desktop/TraFreak/signnames.csv', 'r'))
    d = {}
    for row in reader:
        k, v = row
        d[k] = v
    #graph = tf.get_default_graph()
    model = load_model('C:/Users/Cline/Desktop/TraFreak/traffic_modelnew12.h5')
    #model=load_model('traffic_modelnew.h5')
    img = cv2.imread(destination)
    #plt.imshow(img, cmap=plt.get_cmap('gray'))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    #plt.imshow(img, cmap = plt.get_cmap('gray'))
    print(img.shape)
    img = img.reshape(1, 32, 32, 1)
    ktemp = model.predict_classes(img)
    k = str((*ktemp))
    filepredicted = str(d[k])
    return filepredicted

@app.route("/sign",methods=["POST"])
def signupload():
    target=os.path.join(APP_ROOT,"images/")
    print(target)
	
    if not os.path.isdir(target):
        os.mkdir(target)
	
    for file in request.files.getlist("file"):
        print(file)
        filename=file.filename
        destination="/".join([target,filename])
        print(destination)
        file.save(destination)
        filepred = signpredict(destination)
        
    return render_template("signcomplete.html",image_name=filepred,image=filename)


def vehiclepredict(destination):
    global count
    grid = joblib.load('C:/Users/Cline/Desktop/TraFreak/supportvector.pkl')
    test_image = mpimg.imread(destination)
    #plt.imshow(test_image)
    test_image = test_image.astype(np.float32)/255
    h_start = 100
    h_stop = 480
    
    pixels_in_cell = 16
    HOG_orientations = 11
    cells_in_block = 2
    cells_in_step = 3 
    
    masked_region = test_image[h_start:h_stop,:,:]
    #plt.imshow(masked_region)
    masked_region.shape
    
    resizing_factor = 2
    masked_region_shape = masked_region.shape
    L = masked_region_shape[1]/resizing_factor
    W = masked_region_shape[0]/resizing_factor
    
    masked_region_resized = cv2.resize(masked_region, (np.int(L), np.int(W)))
    masked_region_resized_R = masked_region_resized[:,:,0]
    
    print(masked_region_resized.shape)
    #plt.imshow(masked_region_resized)
    
    masked_region_hog_feature_all, hog_img = hog(masked_region_resized_R, orientations = 11, pixels_per_cell = (16, 16), cells_per_block = (2, 2), transform_sqrt = False, visualize = True, feature_vector = False)
    
    n_blocks_x = (masked_region_resized_R.shape[1] // pixels_in_cell)+1  
    n_blocks_y = (masked_region_resized_R.shape[0] // pixels_in_cell)+1
    
    #nfeat_per_block = orientations * cells_in_block **2 
    blocks_in_window = (64 // pixels_in_cell)-1 
        
    steps_x = (n_blocks_x - blocks_in_window) // cells_in_step
    steps_y = (n_blocks_y - blocks_in_window) // cells_in_step
    
    rectangles_found = []
    
    for xb in range(steps_x):
        for yb in range(steps_y):
            y_position = yb*cells_in_step
            x_position = xb*cells_in_step
                
            hog_feat_sample = masked_region_hog_feature_all[y_position : y_position + blocks_in_window, x_position : x_position + blocks_in_window].ravel()
            x_left = x_position * pixels_in_cell
            y_top = y_position * pixels_in_cell
            print(hog_feat_sample.shape)  
            
            # predict using trained SVM
            #test_prediction = svc_model.predict(hog_feat_sample.reshape(1,-1))
            test_prediction = grid.predict(hog_feat_sample.reshape(1,-1))
            
            if test_prediction == 1: 
                rectangle_x_left = np.int(x_left * resizing_factor)
                rectangle_y_top = np.int(y_top * resizing_factor)
                window_dim = np.int(64 * resizing_factor)
                rectangles_found.append(((rectangle_x_left, rectangle_y_top + h_start),(rectangle_x_left + window_dim, rectangle_y_top + window_dim + h_start)))
             
    Image_with_Rectangles_Drawn = np.copy(test_image)
        
    for rectangle in rectangles_found:
        cv2.rectangle(Image_with_Rectangles_Drawn, rectangle[0], rectangle[1], (0, 255, 0), 20)
   
    
    return Image_with_Rectangles_Drawn,len(rectangles_found)
    
def makename():
    count = 0
    count =  random.randrange(20, 50, 3)
    str1 = 'svm' + str(count) + '.png'
    return str1

@app.route("/vehicle.html",methods=["POST"])
def vehicleupload():    
    target=os.path.join(APP_ROOT,"images/")
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
	
    for file in request.files.getlist("file"):
        print(file)
        filename=file.filename
        destination="/".join([target,filename])
        print(destination)
        file.save(destination)
	
    Image_with_Rectangles_Drawn,carno=vehiclepredict(destination)
    plt.imshow(Image_with_Rectangles_Drawn)
    plt.savefig("images/results/svm.png",dpi=200)
    return render_template("vehiclecomplete.html",image=makename(),cars=carno)

def makename():
    global count
    if count<0:
        count=0
    count = count+1
    str1 = str(count) + '.jpg'
    return str1


def lanepredict(destination):
    rm_jpg()
    rm_mp4()
    directory = r'C:\Users\Cline\Desktop\TraFreak\images\results'
    os.chdir(directory)
    cap = cv2.VideoCapture(destination)
    while(cap.isOpened()):
        try:
            _, frame = cap.read()
            canny_image = canny(frame)
            cropped_canny = region_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            cv2.imshow("result", combo_image)
            cv2.imwrite(makename(),combo_image)
        except TypeError:
            break
    cap.release()
    cv2.destroyAllWindows()
    

    
def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
    
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny

def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    triangle = np.array([[
    (200, height),
    (550, 250),
    (1100, height),]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image
    
      

@app.route("/lane.html",methods=["POST"])
def laneupload():
    target=os.path.join(APP_ROOT,"images/")
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
	
    for file in request.files.getlist("file"):
        print(file)
        filename=file.filename
        destination="/".join([target,filename])
        print(destination)
        file.save(destination)
        rm_jpg()
        rm_mp4()
        lanepredict(destination)
        subprocess.Popen([ "C:/xampp/htdocs/ViewSter/ffmpeg/windows/ffmpeg.exe", '-r', '30', '-start_number', '0', '-i',  os.path.join('C:/Users/Cline/Desktop/TraFreak/images/results/', "%d" + '.jpg'), '-c:v', 'libx264', '-vf', "fps=25,format=yuv420p", os.path.join('C:/Users/Cline/Desktop/TraFreak/images/results/', 'project.mp4') ])
        time.sleep(7)
    return render_template("lanecomplete.html",videourl='results/project.mp4')

if __name__=="__main__":
	app.run(port=4555,debug=True)