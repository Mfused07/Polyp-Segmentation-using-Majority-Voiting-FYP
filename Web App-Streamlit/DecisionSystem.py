from cProfile import label
from re import sub
from tkinter.tix import CELL
import streamlit as st
# import streamlit_authenticator as stauth
import pyrebase
import streamlit as st
from google.cloud import firestore
from fpdf import FPDF
import base64
import numpy as np
import cv2
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from tensorflow.keras import backend as K
import os
import tensorflow as tf
from pdfrw import PageMerge, PdfReader, PdfWriter
from datetime import datetime, date, time
from streamlit import session_state
from PIL import Image,ImageDraw
import io
from io import BytesIO
import collections
import time
import random
try:
    from collections import abc
    collections.MutableMapping = abc.MutableMapping
except:
    pass
# Initially define your users’ names, usernames, and plain text passwords.



firebaseConfig = {
    'apiKey': "AIzaSyAHestWnjbvcTppYjRxOMkItpi5KEhxQ7M",
    'authDomain': "decisionsys-streamlit.firebaseapp.com",
    'databaseURL': "https://decisionsys-streamlit-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'projectId': "decisionsys-streamlit",
    'storageBucket': "decisionsys-streamlit.appspot.com",
    'messagingSenderId': "1006840543134",
    'appId': "1:1006840543134:web:720ca6c6642c18d085088f",
    'measurementId': "G-0YY61YT5NN"
}

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()
st.sidebar.title("Decision System For Polyp Detection")

# Authentication
choice = st.sidebar.selectbox('login/Signup', ['Login', 'Sign up'])


# Obtain User Input for email and password
email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password',type = 'password')


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

# smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


# App 
# Sign up Block

if choice == 'Sign up':
    handle = st.sidebar.text_input(
        'Please input your app handle name', value='Default')
    submit = st.sidebar.button('Create my account')

    if submit:
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account is created suceesfully!')
        st.balloons()
        # Sign in
        # user = auth.sign_in_with_email_and_password(email, password)
        # db.child(user['localId']).child("Handle").set(handle)
        # db.child(user['localId']).child("ID").set(user['localId'])
        # st.title('Welcome' + handle)
        # st.success("Login Successfull")
        # st.info('Login via login drop down selection')

# Login Block
if choice == 'Login':
    login = st.sidebar.checkbox('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email,password)

        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        bio = st.radio('JUMP TO',['Home','Records', 'Settings'])
        
# SETTINGS PAGE 
        if bio == 'Settings':  

            st.write("Firebase Timeout")
  
        elif bio == 'Home':



            
            st.write("""
            # Decision System for Polyp detection
            """
            )

            st.write(""" ### Upload an Endoscopic Image """)

            
            #upload image
            file = st.file_uploader("", type=["jpg", "png"],accept_multiple_files=False)
         
            #create a button
            run = st.button("Generate Mask")

            if(file):
                st.image(file, caption ="Polyp image", width = 400)
            

                #if button is pressed
                my_bar = st.progress(0)
                if run: 
                    for i in range(1,100):
                        time.sleep(0.01)
                        my_bar.progress(i + 1)
                    with CustomObjectScope({'iou': iou}):

                        path = "C:/Users/Mubiii/Desktop/DecisionSystem/model"
                        model = tf.keras.models.load_model(f"{path}/resunet++.h5")
                        filePath = f"C:/Users/Mubiii/Desktop/DecisionSystem/testImages/{file.name}"
                   

                        if 'inputImage' not in st.session_state:
                            st.session_state.inputImage= filePath
                        
                        
                        x = read_image(filePath)

                        y_pred = np.stack([model.predict(np.expand_dims(x, axis=0)) [0] > 0.5 ])
                        h, w, _ = x.shape
                        white_line = np.ones((h, 10, 3)) * 255.0
                                            

                        all_images = [ x , white_line, mask_parse(y_pred) * 255.0 ]

                        imag = np.concatenate(all_images, axis=1)
                        # cv2.imwrite(f"results/{i}.png", image)

                        output = mask_parse(y_pred) * 255.0
                

                        my_image = output
                        

                        st.image(imag, caption=f"Image and Predicted Mask",clamp=True, channels='RBG')
                        image = output

                        faux_file = BytesIO()
                        path = 'C:/Users/Mubiii/Desktop/DecisionSystem/result'
                        x = random.randint(0,1000)
                        cv2.imwrite(os.path.join(path , f'mask{file.name}.jpg'), image)
                        if 'no' not in st.session_state:
                            st.session_state.no = file.name
                
            else:
                st.write("no file boo")



        elif bio == "Records":

            form = st.form(key='my_form')
            patientName = form.text_input(label='Patient Name')
            dob = form.text_input(label='Date of Birth')
            dop = form.text_input(label='Date of Procedure')

            docName = form.text_input(label='Doctor Name')
            Medicines = form.text_input(label='Medications')
            Area = form.text_input(label='Area of Examination')
            visual = form.text_input(label='Visual Quality')
            remarks = form.text_input(label='Doctor Remarks')

            submit = form.form_submit_button(label='Save Record')
 
            db = firestore.Client.from_service_account_json("key.json")


            # Once the user has submitted, upload it to the database
            my_bar = st.progress(0)
            if patientName and dob and docName and Medicines and Area and visual and remarks and submit:
                for i in range(1,100):
                    time.sleep(0.01)
                    my_bar.progress(i + 1)
                sign = ""
                l = [patientName,dob,dop,docName,Medicines,Area,visual, remarks, sign ]
            # Create a reference 
                doc_ref = db.collection("record").document("patient")

                doc_ref.set({
                "name": patientName,
                "dob": dob,
                "dop": dop,
                "docName": docName,
                "Medicines": Medicines,
                "Area": Area,
                "visual": visual,
                "remarks": remarks,

                })
                st.success("Record Successfully Saved")
 
                rep = [
                "Patient Name: ",
                "Date of Birth: ",
                "Date of Procedure:", 
                "Endoscopist: ",
                "Medications: ",
                "Extent of Exam: ",
                "Visualization: ",
                "Remarks: ",
                "Signature:" ]
                heading = "Endoscopy Report"
                pdf = FPDF()
                pdf.add_page()

                # add logo to pdf
                pdf.image("fastf.png",10,8,80)


                



                # add input image polyp


                pdf.image(st.session_state.inputImage,140,40,60)
                f = f"C:/Users/Mubiii/Desktop/DecisionSystem/result/mask{st.session_state.no}.jpg"
                pdf.image(f,140,110,60)



                pdf.set_font('Helvetica', 'B', 16)

                # heading for pdf
                pdf.cell(180, 10, txt = heading,
                        ln = 1, align = 'R')
                # add another cell
                pdf.cell(180, 5, txt = "Decision System",
                        ln = 2, align = 'R')

                pdf.set_draw_color(0, 0, 0) #black line
                pdf.line(5, 30,205 ,30) #coordinate sequence: (x_start, y_start, x_end, y_end)
                pdf.set_font('Helvetica', '', 10)

                # Patient Details
                
                # pdf.cell(200, 20, txt = "Image",ln = 2, align = 'L')
                for i in range(len(rep)):
                    pdf.cell(200, 20, txt = rep[i]+l[i],ln = 2, align = 'L')

                html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
                st.markdown(html, unsafe_allow_html=True)
                

                # And then render each record, using some light Markdown

           
            # def foo():
            #     st.write("loading Records")
            #     doc_ref = db.collection(u'record').document(u'patient')
            #     doc = doc_ref.get()

            #     if doc.exists:
            #         st.write(f'Document data: {doc.to_dict()}')
            #         name = doc.to_dict()["name"]
            #         dob = doc.to_dict()["dob"]
            #         dob = doc.to_dict()["dop"]
            #         docName = doc.to_dict()["docName"]
            #         Medicines = doc.to_dict()["Medicines"]
            #         Area = doc.to_dict()["Area"]
            #         visual = doc.to_dict()["visual"]
            #         remarks = doc.to_dict()["remarks"]
            #         sign = ""
     
        
            #     else:
            #         st.write(u'No such document!')



            # showRecord = form.form_submit_button(label='showRecord')
            # if showRecord:

            #     foo()


            
            