import tkinter as tk
from tkinter import filedialog
from tkinter.messagebox import OK, INFO, showinfo
import cv2
import tensorflow as tf
import numpy as np
import os

from object_detection.utils import visualization_utils as viz_utils

import Program.parameters as params
import Detector.model as model
import Detector.data as data
import Tkinter.tkinter_parameters as tk_params


def build_main_window():

    #just a function which creates a window template with one label

    root = tk.Tk()
    root.title(tk_params.window.title)
    root.geometry(tk_params.window.geometry)

    label_main = tk.Label(root, text=tk_params.window.label_main, font=("Arial", 24))
    label_main.pack(pady=30)

    return root


def show_info_before():
    showinfo(title=tk_params.window.title, message="This process needs some time to run. When it finishes, you'll see an another message", 
                detail="To start the process, click OK", icon=INFO, default=OK)

def show_info_after():
    showinfo(title=tk_params.window.title, message="The process finished successfully", 
                detail="Click OK to continue", icon=INFO, default=OK)


def main_tkinter_menu():
    
    def click_button_custom():
        #!cd {labelImg_path} && python labelImg.py
        root.destroy()
        choose_folder_training()

    def click_button_list():
        print("List click")
    

    root = build_main_window()

    label_2 = tk.Label(root, text="Choose the way to detect your objects", font=("Arial", 20))
    label_2.pack(pady=100)

    button_custom = tk.Button(root, text="Custom detection", command=click_button_custom)
    button_list   = tk.Button(root, text="Select an object from the list", command=click_button_list)

    button_custom.pack(side='left', padx=200)
    button_list.pack(side='left', padx=20)

    root.mainloop()



def choose_folder_training():
    
    def click_button_folder_training():
        
        directory = filedialog.askdirectory()
        if directory != "":

            show_info_before()

            images_np, boxes_np = data.process_training_folder(directory)
            image_tensors, classes_one_hot_tensors, box_tensors = data.prepping_data(images_np, boxes_np)
            
            detection_model, to_fine_tune = model.prepare_model_for_training(
                                        params.settings.pipeline_config,
                                        params.settings.checkpoint_path)
            
            model.train_the_model(images_np, image_tensors, classes_one_hot_tensors,
                            box_tensors, detection_model, to_fine_tune)
            
            show_info_after()

            root.destroy()
            finish_custom_detection(detection_model)
        
        
    root = build_main_window()
    
    label_2 = tk.Label(root, text="Choose the folder for training if you're ready", font=("Arial", 20))
    label_2.pack(pady=100)
    
    button_main = tk.Button(root, text="Choose", command=click_button_folder_training)
    button_main.pack(side='left', padx=400)
    
    root.mainloop()




def finish_custom_detection(detection_model):
    
    def click_button_select_labelled():
        directory = filedialog.askdirectory()
        if directory != "":

            show_info_before()

            data.save_detections_from_folder(directory, params.category_index, detection_model)
            
            show_info_after()

            root.destroy()
            finish_custom_detection(detection_model)
    
    def finish_detection():
        root.destroy()
        view_tkinter_results()
    
    def click_button_camera():
        
        # define a video capture object
        vid = cv2.VideoCapture(0)

        while True:
            # Capture the video frame by frame
            _, frame = vid.read()
            frame = cv2.flip(frame, 1)

            #get detections
            input_tensor = tf.convert_to_tensor(np.expand_dims(frame, axis=0), dtype=tf.float32)
            detections = model.detect(input_tensor, detection_model)

            viz_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                detections['detection_boxes'][0].numpy(),
                detections['detection_classes'][0].numpy().astype(np.uint32) + params.settings.label_id_offset,
                detections['detection_scores'][0].numpy(),
                params.category_index,
                use_normalized_coordinates=True,
                min_score_thresh=params.hyperparameters.min_score_thresh)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # the 'q' button is set as the quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
    
    
    root = build_main_window()
    
    label_2 = tk.Label(root, text="Choose the folder with labelled images or", font=("Arial", 20))
    label_2.pack(pady=100)
    
    label_3 = tk.Label(root, text="get detections in real-time", font=("Arial", 20))
    label_3.pack(pady=10)
    
    button_main = tk.Button(root, text="Select folder", command=click_button_select_labelled)
    button_main.pack(side='left', padx=200)
    
    button_camera = tk.Button(root, text="Real-time", command=click_button_camera)
    button_camera.pack(side='left', padx=20)
    
    button_finish = tk.Button(root, text="Finish", command=finish_detection)
    button_finish.pack(padx=200, pady=20)
    
    root.mainloop()




def view_tkinter_results():
    
    def click_button_view_results():
        os.startfile(r'.\media\out')

    root = build_main_window()
    
    label_2 = tk.Label(root, text="Detecting was finished successfully!", font=("Arial", 20))
    label_2.pack(pady=100)
    
    button_main = tk.Button(root, text="View", command=click_button_view_results)
    button_main.pack(side='left', padx=400)
    
    root.mainloop()

