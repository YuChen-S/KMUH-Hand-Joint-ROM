import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import mediapipe as mp
import cv2
# from google.colab.patches import cv2_imshow
# from streamlit_gsheets import GSheetsConnection

### Functions for page layout ###
def st_website_setting():
    st.set_page_config(
        page_title="KMUH Hand Joint ROM Recognition",
        page_icon=":ok_hand:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def st_sidebar_info():
    with st.sidebar:
        st.title("About")
        st.info(
            """
            A webtool automatically calculates the ROM of each joint hand from 2 photos, oblique flexion view and oblique extension view.\n
            Acclerate the workflow for hand surgeon at outpatient department or pre-operative assessment\n
            """
        )
        st.title("Contact")
        st.info(
            """
            Developer: 
            [Yu Chen Shen](https://www.linkedin.com/in/199707-yuchen-shen/), \n
            Director: 
            [Andy Liu](https://www.linkedin.com/in/andysirliu/)
            """
        )
        st.success("Contact yuchen2690720@gmail.com if any problem be found, thanks")
        st.write('''---''')

def st_title_info():
    st.title('KMUH Hand Joint ROM Calculator 2025')
    st.subheader(
    '''
    This application will help you to accelerate the evaluation of hand joint ROM.
    ''', anchor=None)
    st.markdown(''' **3 main features:** 
    1. Freely assessed just by internet.
    2. Upload photos from local storage or by camera of user device.
    3. Automatically calculate the ROM of all the hand joints.
    ''')
    st.write('**Please check the user guide if any question. Thanks for using our product!**')
    st.write('---')

### Functions for hand landmark recognition ###
def extract_hand_landmarks(hands, image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image_rgb)

    # Check if hand landmarks were detected
    if not results.multi_hand_landmarks:
        raise ValueError(f"No hand landmarks detected in {image_path}.")

    # Extract landmarks
    '''
    Landmark Extraction: 
    Detected landmarks are normalized; they are converted to pixel coordinates based on the image dimensions. 
    The z coordinate remains as provided, representing depth relative to the wrist.
    '''
    hand_landmarks = results.multi_hand_landmarks[0]

    # Convert normalized landmarks to pixel coordinates
    image_height, image_width, _ = image.shape
    landmarks = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        z = landmark.z  # z is already in meters relative to the wrist
        landmarks.append((x, y, z))

    return landmarks, hand_landmarks

def visualize_landmarks(mp_hands, mp_drawing, image_path, hand_landmarks, output_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    # Convert the BGR image to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw the landmarks on the image
    '''
    Uses mp_drawing.draw_landmarks() to overlay detected landmarks on the image.
    Connects the landmarks with lines (connections) to form the hand skeleton.
    Customization: You can modify the color, thickness, and circle radius of the landmarks and connections by adjusting DrawingSpec.
    '''
    mp_drawing.draw_landmarks(
        image, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2)
    )
    # Save the image with landmarks overlay
    cv2.imwrite(output_path, image)
    # Show the image with landmarks overlay
    # cv2.imshow('Your result', image)

# Define a function to calculate the angle between three points
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    # a . b = |a|*|b| * Cos(theta)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Define a function to compute joint angles from landmarks
def compute_joint_angles(landmarks):
    joint_angles = {}
    # Convert landmarks to numpy arrays
    landmarks_np = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    # Define joint connections based on MediaPipe hand landmark indices
    joints = {
        'thumb_cmc': (0, 1, 2),
        'thumb_mcp': (1, 2, 3),
        'thumb_ip': (2, 3, 4),
        'index_mcp': (0, 5, 6),
        'index_pip': (5, 6, 7),
        'index_dip': (6, 7, 8),
        'middle_mcp': (0, 9, 10),
        'middle_pip': (9, 10, 11),
        'middle_dip': (10, 11, 12),
        'ring_mcp': (0, 13, 14),
        'ring_pip': (13, 14, 15),
        'ring_dip': (14, 15, 16),
        'pinky_mcp': (0, 17, 18),
        'pinky_pip': (17, 18, 19),
        'pinky_dip': (18, 19, 20),
    }
    # Calculate angles for each joint
    for joint_name, (a_idx, b_idx, c_idx) in joints.items():
        a = landmarks_np[a_idx]
        b = landmarks_np[b_idx]
        c = landmarks_np[c_idx]
        joint_angles[joint_name] = calculate_angle(a, b, c)
    return joint_angles