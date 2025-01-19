import streamlit as st
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils import *
from PIL import Image

###### webpage settings ######
st_website_setting()

###### sidebar info ######
st_sidebar_info()

###### Main page title and user guide #####
st_title_info()

##### Main block #####

### Initialize MediaPipe Hand Landmarker ###
# The mp.solutions.hands.Hands class is initialized with static_image_mode=True to process static images.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

### Upload block ###
# Let user to upload 2 photos, including hand flexion and extension view
with st.container():
    st.subheader('Please upload your photos')
    photo_col_flexion, photo_col_extension = st.columns(2)
    with photo_col_flexion:
        flexion_photo = st.file_uploader('Please upload the flexion view of your hand', type=['jpg', 'png'], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    with photo_col_extension:
        extension_photo = st.file_uploader('Please upload the extension view of your hand', type=['jpg', 'png'], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    if flexion_photo != None and extension_photo != None:
        display_col_flexion, display_col_extension = st.columns(2)
        with display_col_flexion:
            st.image(flexion_photo)
            flexion_img = Image.open(flexion_photo)
            flexion_img.save('user_hand_flexion.jpg')
        with display_col_extension:
            st.image(extension_photo)
            extension_img = Image.open(extension_photo)
            extension_img.save('user_hand_extension.jpg')            

### Start block ###
submit = st.button("Start Hand Joint ROM Calculation")
show_result = False
with st.container():
    if submit:
        my_bar = st.progress(0)

        for percent_complete in range(0, 100, 10):
             time.sleep(0.02)
             my_bar.progress(percent_complete + 10)
        st.success('Thanks for waiting, please check your result!')
        st.write('---')
        show_result = True

### Image recognition block ###
with st.container():
    if submit:
        # Paths to your images
        flexion_image_path = './user_hand_flexion.jpg'
        extension_image_path = './user_hand_extension.jpg'
        flexion_output_path = './flexion_with_landmarks.jpg'
        extension_output_path = './extension_with_landmarks.jpg'

        # Extract landmarks and visualize/save them
        flexion_landmarks, flexion_hand_landmarks = extract_hand_landmarks(hands, flexion_image_path)
        extension_landmarks, extension_hand_landmarks = extract_hand_landmarks(hands, extension_image_path)

        # Output the landmarks
        # print("Flexion Landmarks:", flexion_landmarks)
        # print("Extension Landmarks:", extension_landmarks)

        # Visualize and save the landmarks overlay images
        visualize_landmarks(mp_hands, mp_drawing, flexion_image_path, flexion_hand_landmarks, flexion_output_path)
        visualize_landmarks(mp_hands, mp_drawing, extension_image_path, extension_hand_landmarks, extension_output_path)
        # display_images = ['flexion_with_landmarks.jpg', 'extension_with_landmarks.jpg']

        # Compute joint angles
        flexion_angles = compute_joint_angles(flexion_hand_landmarks)
        extension_angles = compute_joint_angles(extension_hand_landmarks)

        # Calculate ROM for each joint
        joint_rom = {joint: abs(extension_angles[joint] - flexion_angles[joint]) for joint in flexion_angles}
        df_joint_rom = pd.DataFrame(joint_rom.values(), index = joint_rom.keys(), columns = ['The joint ROM'])
        # Clean up
        hands.close()

### Result block ###
with st.container():
    if show_result:
        output_col_flexion, output_col_extension = st.columns(2)
        with output_col_flexion:
            st.image('flexion_with_landmarks.jpg')
        with output_col_extension:
            st.image('extension_with_landmarks.jpg')
        st.markdown("# The ROM of all the hand joints: ")
        st.table(df_joint_rom)
        st.write(joint_rom)