#!/usr/bin/env python
# coding: utf-8

# # Report 1

# ## Setting up Environments and Import Images

# In[1]:

# In[2]:


import requests
import zipfile
import io
import os
from IPython.display import display, Image, clear_output, HTML
import time
import pandas as pd
import os
import glob
import random
import threading
from PIL import Image
import ipywidgets as widgets
from ipywidgets import VBox, Button, RadioButtons
from fractions import Fraction
import sys
import datetime
from jupyter_ui_poll import ui_events
from PIL import Image as PILImage
import numpy as np
import math

# Step 1: Download the repository as a ZIP file and extract it
url_images = "https://github.com/BohanZhangPRC/ANS-Test/archive/refs/heads/main.zip"
response = requests.get(url_images)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("ANS-Test")

# Step 2: List files in the extracted directory
extracted_path = r"ANS-Test\ANS-Test-main"
#for root, dirs, files in os.walk(extracted_path):
    #print('Images downloaded:')
    #for name in files:
        #print(os.path.join(root, name))
        
url_white = "https://github.com/BohanZhangPRC/Others/archive/refs/heads/main.zip"
response = requests.get(url_white)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("Others")

white_path = r"Others"
#for root, dirs, files in os.walk(white_path):
    #for name in files:
        #print(os.path.join(root, name))
        
clear_output(wait=0)
# ## User Consent and Info

# In[3]:


data_consent_info = """DATA CONSENT INFORMATION:

Please read:

we wish to record your response data
to an anonymised public data repository. 
Your data will be used for educational teaching purposes
practising data analysis and visualisation.

Please type   yes   in the box below if you consent to the upload."""
def consent():
    print(data_consent_info)
    result = input("> ").strip().lower()  # Normalize the input to lowercase

    if result in ["yes"]:  # Check if the normalized input is in the list of acceptable values
        print("Thanks for your participation.")
        print("Please contact bohan.zhang.21@ucl.ac.uk")
        print("If you have any questions or concerns")
        print("regarding the stored results.")
    
    else:
        raise(Exception("User did not consent to continue test."))
        sys.exit()  # Exit the script if user did not consent

id_instructions = """

Enter your anonymised ID

To generate an anonymous 4-letter unique user identifier please enter:
- two letters based on the initials (first and last name) of a childhood friend
- two letters based on the initials (first and last name) of you

e.g. if your friend was called Charlie Brown and you are Tom Cruise
     then your unique identifer would be CBTC
"""
def get_id():
    print(id_instructions)
    user_id = input("> ")
    print("User entered id:", user_id)
    return user_id

import time
def initiate():
    consent()
    user_id = get_id()
    return user_id
    

user_id = initiate()
clear_output(wait=1)


# ## Create test images

# In[4]:


# Define the root directory to start the search

if os.listdir(extracted_path):
    root_directory = extracted_path

else: 
    print("Please enter your path of images.")
    root_directory = input(r"")

image_files = glob.glob(os.path.join(root_directory, "*.png"))
if image_files:
    if len(image_files) == 64:
        print(f"Found {len(image_files)} '.png' files, Expect 64 '.png' files.")
        for file in image_files:
            print(file)
        
        clear_output(wait = False)
        #print("Number of images is correct, proceed the test.")
    else:
        raise(Exception("Number of images is incorrect, please check if any of the image files is missing."))
else:
    raise(Exception("No .png files found in the specified directory. Please check the directory"))

clear_output(wait = True)   

image_data = []

for image_file in image_files:
    # Extract the filename without the path
    file_name = os.path.basename(image_file).replace('.png', '')
    
    # Remove the file extension and split the name by '-'
    parts = file_name.split('-')
    
    # Extract the first color and value
    color_left = parts[0][0]  # First character, e.g., 'B'
    value_left = int(parts[0][1:])  # The numeric value, e.g., 10
    
    # Extract the second color and value
    color_right = parts[1][0]  # First character, e.g., 'Y'
    value_right = int(parts[1][1:])  # The numeric value, e.g., 9
    
    # Append to the data list
    image_data.append({
        'image_file': image_file,
        'file_name': file_name,
        'ColorLeft': color_left,
        'ValueLeft': value_left,
        'ColorRight': color_right,
        'ValueRight': value_right
    })

# Create a DataFrame
df_images = pd.DataFrame(image_data)


# ## Test Codes

# In[17]:


# Randomize Images
df_images = df_images.sample(frac=1).reset_index(drop=True)

# Initialize an empty list for test results
test_results = []

# Load your DataFrame (assuming df_images is already loaded)
# df_images = pd.read_csv('your_csv_file.csv')

# Set up widgets
start_button = widgets.Button(description="Start Test")
image_widget = widgets.Output()
question_label = widgets.Label(value="Which side has more dots?")
options = widgets.RadioButtons(options=['Left', 'Right'], layout={'width': '1000px'})  # Choose between left and right
submit_button = widgets.Button(description="Submit")
out = widgets.Output()
display(out)

# State variables
test_started = False
current_row = None
test_index = 0 

event_info = {}

# Function to wait for an event (button press, etc.)
def wait_for_event(timeout=-1, interval=0.001, max_rate=20, allow_interrupt=True):    
    start_wait = time.time()
    event_info['type'] = ""
    event_info['description'] = ""
    event_info['time'] = -1

    n_proc = int(max_rate*interval) + 1
    
    with ui_events() as ui_poll:
        keep_looping = True
        while keep_looping:
            ui_poll(n_proc)

            if (timeout != -1) and (time.time() > start_wait + timeout):
                keep_looping = False
                
            if allow_interrupt and event_info['description'] != "":
                keep_looping = False

            time.sleep(interval)
    
    return event_info

# Function to handle button events
def register_btn_event(btn):
    event_info['type'] = "button click"
    event_info['description'] = btn.description
    event_info['time'] = time.time()
    return
# Function to display the next image
def display_next_image(index):
    if index < len(df_images):
        row = df_images.iloc[index]
        img = PILImage.open(row['image_file'])
        white = PILImage.open('Others\Others-main\White.jpg')
        white = white.resize((500, int(img.height * 500 / img.width)))
        img = img.resize((500, int(img.height * 500 / img.width)))

        with image_widget:
            display(img)
            wait_for_event(timeout=0.75)
            clear_output(wait=False)
            display(white)
            clear_output(wait=True)
        
        return row
    else:
        # Test is complete
        with out: clear_output(wait=False)
        with out: print("Test completed.")
        with out: print("Thank You!")
        # Trigger result generation here
        generate_results()  # Call the function to generate results
        return None
# Function to handle button click for starting the test
def on_start_button_click(b):
    global test_started, current_row, test_index
    if not test_started:
        test_started = True
        clear_output(wait=False)
        with out: display(widgets.VBox([image_widget, question_label, options, submit_button]))
        current_row = display_next_image(test_index)
        submit_button.on_click(on_submit_button_click)

# Function to handle button click for submission
def on_submit_button_click(b):
    global current_row, test_index
    
    if options.value:
        correct_answer = 'Left' if current_row['ValueLeft'] > current_row['ValueRight'] else 'Right'
        ratio = Fraction(current_row['ValueRight'], current_row['ValueLeft'])
        if current_row['ValueLeft'] > current_row['ValueRight']: 
            r = current_row['ValueRight'] / current_row['ValueLeft']
        else:
            r = current_row['ValueLeft'] / current_row['ValueRight']
        test_results.append({
            'file_name': current_row['file_name'],
            'ColorLeft': current_row['ColorLeft'],
            'ValueLeft': current_row['ValueLeft'],
            'ColorRight': current_row['ColorRight'],
            'ValueRight': current_row['ValueRight'],
            'user_choice': options.value,
            'correct_answer': correct_answer,
            'is_correct': options.value == correct_answer,
            'ratio': f"{current_row['ValueLeft']}:{current_row['ValueRight']}",
            'simplified_ratio': f"{ratio.numerator}:{ratio.denominator}",
            'r': r
        })
        test_index += 1
        current_row = display_next_image(test_index)

        options.value = None

# Function to generate results after test completion
def generate_results():
    #check the results
    df_results = pd.DataFrame(test_results)
    #test_results
    #df_results

    # Define the ratios you want to find
    ratios_8_9 = ["9:8", "8:9"]
    ratios_9_10 = ["9:10", "10:9"]
    ratios_6_7  = ["6:7", "7:6"]
    ratios_3_4 = ["3:4", "4:3"]

    # Filter the DataFrame to include only rows with these ratios
    r_8_9 = df_results[df_results['simplified_ratio'].isin(ratios_8_9)]
    r_9_10 = df_results[df_results['simplified_ratio'].isin(ratios_9_10)]
    r_6_7 = df_results[df_results['simplified_ratio'].isin(ratios_6_7)]
    r_3_4 = df_results[df_results['simplified_ratio'].isin(ratios_3_4)]
    # Filter the DataFrame to include only rows with these ratios and same color
    r_3_4_s = df_results[(df_results['simplified_ratio'].isin(ratios_3_4)) & (df_results['ColorRight'] == df_results['ColorLeft'])]
    r_6_7_s = df_results[(df_results['simplified_ratio'].isin(ratios_6_7)) & (df_results['ColorRight'] == df_results['ColorLeft'])]
    r_8_9_s = df_results[(df_results['simplified_ratio'].isin(ratios_8_9)) & (df_results['ColorRight'] == df_results['ColorLeft'])]
    r_9_10_s = df_results[(df_results['simplified_ratio'].isin(ratios_9_10)) & (df_results['ColorRight'] == df_results['ColorLeft'])]
    # Filter the DataFrame to include only rows with these ratios and different color
    r_3_4_d = df_results[(df_results['simplified_ratio'].isin(ratios_3_4)) & (df_results['ColorRight'] != df_results['ColorLeft'])]
    r_6_7_d = df_results[(df_results['simplified_ratio'].isin(ratios_6_7)) & (df_results['ColorRight'] != df_results['ColorLeft'])]
    r_8_9_d = df_results[(df_results['simplified_ratio'].isin(ratios_8_9)) & (df_results['ColorRight'] != df_results['ColorLeft'])]
    r_9_10_d = df_results[(df_results['simplified_ratio'].isin(ratios_9_10)) & (df_results['ColorRight'] != df_results['ColorLeft'])]

    # Calculate the error rate for same color with each ratio
    e_3_4_s = 1 - r_3_4_s['is_correct'].mean()
    e_6_7_s = 1 - r_6_7_s['is_correct'].mean()
    e_8_9_s = 1 - r_8_9_s['is_correct'].mean()
    e_9_10_s = 1 - r_9_10_s['is_correct'].mean()

    # Calculate the error rate for different color with each ratio
    e_3_4_d = 1 - r_3_4_d['is_correct'].mean()
    e_6_7_d = 1 - r_6_7_d['is_correct'].mean()
    e_8_9_d = 1 - r_8_9_d['is_correct'].mean()
    e_9_10_d = 1 - r_9_10_d['is_correct'].mean()

    same_colors_df = df_results[df_results['ColorLeft'] == df_results['ColorRight']]
    different_colors_df = df_results[df_results['ColorLeft'] != df_results['ColorRight']]

    # Save the Data
    t = f'{datetime.datetime.now()}'
    t = t.replace(" ", "-")
    t = t.replace(":", "-")
    t = t[0:16]
    df_results.to_csv(f'{user_id}-{t}.csv', header=True)
    
    def get_Pe(r, w):
        if w > 0:
            # Calculation for w > 0
            numerator = 1 - r
            denominator = math.sqrt(r**2 + 1) * w * math.sqrt(2)
            erfc_argument = numerator / denominator
            Pe_value = 0.5 * math.erfc(erfc_argument)
        else:
            # When w = 0, Pe(r) = 0
            Pe_value = 0

        return Pe_value

    def get_SSR(e_3_4, e_6_7, e_8_9, e_9_10, w):
        # Ratios corresponding to the given error rates
        ratios = [3/4, 6/7, 8/9, 9/10]

        # Given error rates
        observed_errors = [e_3_4, e_6_7, e_8_9, e_9_10]

        # Calculate the predicted error rates using Pe(r, w)
        predicted_errors = [get_Pe(r, w) for r in ratios]

        # Calculate residuals (observed - predicted), square them, and sum them up
        SSR = sum((obs - pred) ** 2 for obs, pred in zip(observed_errors, predicted_errors))

        return SSR

    # Create an array of w values from 0 to 1 in steps of 0.01
    w_values = np.arange(0, 1.01, 0.01)

    # Calculate SSR for each w value
    ssr_s = [get_SSR(e_3_4_s, e_6_7_s, e_8_9_s, e_9_10_s, w) for w in w_values]
    ssr_d = [get_SSR(e_3_4_d, e_6_7_d, e_8_9_d, e_9_10_d, w) for w in w_values]

    # Find the w with the minimum SSR of same color
    min_ssr_index_s = np.argmin(ssr_s)
    best_w_s = w_values[min_ssr_index_s]
    min_ssr_s = ssr_s[min_ssr_index_s]

    # Find the w with the minimum SSR of different color
    min_ssr_index_d = np.argmin(ssr_d)
    best_w_d = w_values[min_ssr_index_d]
    min_ssr_d = ssr_s[min_ssr_index_d]


    print(f"The value of w that minimizes the SSR for same color is: {best_w_s:.2f}")
    print(f"Minimum SSR value for same color: {min_ssr_s:.4f}")
    print(f"The value of w that minimizes the SSR for different color is: {best_w_d:.2f}")
    print(f"Minimum SSR value for different color: {min_ssr_d:.4f}")

    # Calculate the error rate
    calculations = []
    t = f'{datetime.datetime.now()}'
    t = t[0:16]

    calculations.append({'user_id': user_id,
                         'best w for same colour': best_w_s,
                         'best w for different colour': best_w_d,
                         'time': f'{t}'
                        })  # Mean of 1's and 0's gives the proportion of correct entries
    df_calc = pd.DataFrame(calculations)
    df_calc
    # File name
    filename = 'result_w.csv'

    # Save the Weber values
    if os.path.exists(filename):
        # Read the existing CSV file into a DataFrame
        df_all_subjects = pd.read_csv(filename)
        df_all_subjects = pd.concat([df_all_subjects, df_calc], axis = 0)
        df_all_subjects.to_csv(filename, index=False)
        print("File read successfully.")
    else:
        # Save the DataFrame to a CSV file
        df_calc.to_csv(filename, index=False)
        print("File created and default data saved.")


# Initial display of the start button
with out: display(start_button)
with out: start_button.on_click(on_start_button_click)
