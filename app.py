import os
import flask
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from flask_sqlalchemy import SQLAlchemy
import datetime 
import sqlite3
import time
import re
from flask_cors import cross_origin, CORS

UPLOAD_FOLDER = '/flasker/static'
ALLOWED_EXTENSIONS = {'mp4'}

frames_arr = []
analyzed_arr=[]
commit_arr = []
final_arr=[]

app = flask.Flask(__name__)
CORS(app, support_credentials=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY']= 'test'

conn = sqlite3.connect('plates.db', check_same_thread=False)
c= conn.cursor()

def dynamic_entry(plates, filename, videoName):
    temp = re.findall(r'\d+', str(filename))
    res = list(map(int, temp))
    conversion = str(datetime.timedelta(seconds=res[0]))
    unix = time.time()
    date = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S'))
    plate= plates
    c.execute("INSERT INTO allPlates (unix, datestamp, plate, videoTs, videoName) VALUES (?,?,?,?,?)",
        (unix, date, plate, conversion, videoName))
    conn.commit()

def read_photo(photo, filename, videoName):
    print("READ PHOTO")
    img = cv2.imread(photo)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11,17,17)
    edged = cv2.Canny(bfilter, 30,200)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255,-1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    try:
        text = result[0][-2]      
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        cv2.imwrite("selected/"+str(filename), res)
        temp=[]
        temp.append(text)
        temp.append(filename)
        temp.append(videoName)
        analyzed_arr.append(temp)        
    except:
        print('n a mers boz')
    else:
        cv2.imwrite("selected/"+str(filename), res)
        temp=[]
        temp.append(text)
        temp.append(filename)
        temp.append(videoName)
        analyzed_arr.append(temp)
    finally:
        cv2.imwrite("selected/"+str(filename) , res) 
        temp=[]
        temp.append(text)
        temp.append(filename)
        temp.append(videoName)
        analyzed_arr.append(temp)

def video_to_frames(video):
    print("VIDEO TO FRAMES", video)
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_table():
    c.execute("SELECT * FROM allPlates")
    conn.commit()
    rows = c.fetchall()
    line = ""
    for i in rows:
        line += "<tr> <td>" + str(i[0]) + "</td> <td>" + str(i[1]) + "</td> <td>" + str(i[2]) + "</td> <td>" + str(i[3]) + "</td> <td>" + str(i[4]) + "</td> <tr>"
    return(line)

test = ''
@cross_origin()
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        print('POST MEHTOD')
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print("FILE:",file)
            filename = file.filename
            test = str(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            video_to_frames("static/" + filename)
            directory = 'frames'
            for filename in os.listdir(directory):
                print("FILENAME:",filename)
                f = os.path.join(directory, filename)
                print("For method for:",filename)
                print(f)
                if os.path.isfile(f):       
                    temp_arr=[]
                    temp_arr.append(f)
                    temp_arr.append(filename)
                    temp_arr.append(str(file.filename))
                    frames_arr.append(temp_arr)
                    print(len(frames_arr))
            for i in frames_arr:
                try:
                    read_photo(i[0], i[1], i[2]) 
                    
                except:
                    print('could not read photo')
            
            for i in analyzed_arr:
                if i[0] not in commit_arr:
                    commit_arr.append(i[0])
                    final_arr.append(i)
                    print(commit_arr)
                    
            for i in final_arr:
                dynamic_entry(i[0], i[1], i[2])
            
            #(plates, filename, videoName
            #'BABMEC', 'frame1.jpg', 'video2.mp4']

    return '''
    <h1>DONE</h1>
    '''
    

@app.route('/results') 
def results():
    print('?')
    return '''
    <h1>Number Plates</h1>  
    <video width="320" height="240" controls>
    <source src="/static/'''+'video2.mp4'+ '''"type="video/mp4">
    Your browser does not support the video tag.
    </video>
    '''\
    "<table><tr><th>unix </th><th>datestamp </th><th>plate </th><th>videoTs </th><th>videoName </th></tr>" + add_table() +"</table>"
   
@app.route('/test')   
def test():
    return '''
    <h1>muie?</h1>
    ''' 
