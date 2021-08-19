import tkinter as tk
from tkinter import messagebox
from tkinter import PhotoImage, Label
import json
import cv2
from PIL import Image, ImageTk
import winsound
from DrowsyData import *
from datetime import *
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import shutil

TITLE_FONT = ("Verdana", 12)
LARGE_FONT = ("Verdana", 12)
SMALL_FONT = ("Verdana", 8)

nocap_WIDTH = 325
nocap_HEIGHT = 250

window_icon = "Images/zzz_icon.png"

background_COLOR = '#59A300'
background_IMAGE = "Images/nocap_background.png"
cap_background_IMAGE = "Images/cap_background.png"

scatter_sample = "Images/scatterplot_sample.png"
histogram_sample = "Images/histogram_sample.png"
pie_sample = "Images/pie_sample.png"
table_sample = "Images/table_sample.png"

text_COLOR = 'black'
label_COLOR = 'lightblue'

button_COLOR = 'lightblue'
button_WIDTH = 10
button_pady = 3

cap_WIDTH = 650
cap_HEIGHT = 610

settings_filename = 'settings.json'
data_filename = 'data.csv'
export_folder_name = 'exports'

# xml files of face and eye cascade classifiers
FACE_CASCADE = 'cascades/haarcascade_frontalface_default.xml'
EYE_CASCADE = 'cascades/haarcascade_eye_tree_eyeglasses.xml'

DURATION = 500  # Beep duration in ms
FREQ = 600      # Beep frequency in Hz


class Main(tk.Tk):
    def __init__(self, *args, **kargs):
        # Initialize TKinter Window
        tk.Tk.__init__(self, *args, **kargs)
        container = tk.Frame(self)
        self.title('Drowsiness Detector')
        self.iconphoto(False, PhotoImage(file=window_icon))

        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        self.previous_window = None

        for F in (MainWindow, PreviewWindow, RecordingWindow, AlertWindow, SettingsWindow, ResultsWindow):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(MainWindow)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        frame.start()

    def set_previous_window(self, window):
        self.previous_window = window

    def get_previous_window(self):
        return self.previous_window

    @staticmethod
    def fix_corruption():
        settings_file = open(settings_filename, "w+")
        default_settings = {"eye_thresh": 3,
                            "alarm_thresh": 10,
                            "snooze_thresh": 10}
        settings_file.write(json.dumps(default_settings, indent=4))
        settings_file.close()

    def get_settings(self):
        try:
            file = open(settings_filename, "r")
            data = json.loads(file.read())
            file.close()
            return data
        except FileNotFoundError:
            self.fix_corruption()
            file = open(settings_filename, "r")
            data = json.loads(file.read())
            file.close()
            return data
        except json.decoder.JSONDecodeError:
            tk.messagebox.showerror(
                "ERROR", "File Corruption detected!\n\nReverting 'settings.json' to default values")
            self.fix_corruption()
            file = open(settings_filename, "r")
            data = json.loads(file.read())
            file.close()
            return data

    # Create a datafile wby opening with the "write" tag "time,duration,type\n"
    # will be written to the top of this file as they are the currently used columns in our data
    @staticmethod
    def reset_datafile():
        data_file = open('data.csv', 'w')
        data_file.write("time,duration,type\n")
        data_file.close()


class MainWindow(tk.Frame):
    def __init__(self, parent, controller):
        self.controller = controller

        # Initialize Tkinter frame
        tk.Frame.__init__(self, parent)

        background_image = tk.PhotoImage(file=background_IMAGE)
        BGlabel = tk.Label(self, image=background_image)
        BGlabel.image = background_image
        BGlabel.place(x=0, y=0, width=nocap_WIDTH, height=nocap_HEIGHT)

        label = tk.Label(
            self, text=" Welcome to the Drowsiness Detector ", font=LARGE_FONT, bg=label_COLOR)
        label.pack(pady=10, padx=10)

        run_button = tk.Button(self, text="Run", bg=button_COLOR, width=button_WIDTH,
                               command=lambda: controller.show_frame(PreviewWindow))
        run_button.pack(pady=button_pady)

        end_button = tk.Button(self, text="End Program", bg=button_COLOR, width=button_WIDTH,
                               command=lambda: self.quit())
        end_button.pack(pady=button_pady)

    def start(self):
        self.controller.geometry("{}x{}".format(nocap_WIDTH, nocap_HEIGHT))
        self.controller.reset_datafile()


class PreviewWindow(tk.Frame):
    def __init__(self, parent, controller):
        self.controller = controller

        tk.Frame.__init__(self, parent)

        background_image = tk.PhotoImage(file=cap_background_IMAGE)
        BGlabel = tk.Label(self, image=background_image)
        BGlabel.image = background_image
        BGlabel.place(x=0, y=0, width=cap_WIDTH, height=cap_HEIGHT)

        notice_label = tk.Label(
            self, text="Check face and eye detection positions", bg=label_COLOR, font=LARGE_FONT)
        notice_label.pack()

        # Initialize face and eye cascade classifiers from cv2 imported xml files
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE)

        # Initialize video capture member variable with a cv2 video capture and immediately release.
        # Although there is no visual effect, this initialization and subsequent release of the video capture
        # resolves a graphical error that occurs when switching to previewWindow for the first time
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture.release()
        cv2.destroyAllWindows()

        # Frame creation for webcam view to be placed in
        self.preview_frame = tk.Label(self)
        self.preview_frame.pack(pady=10, padx=10)

        # Button leading to the Settings page
        settings_button = tk.Button(
            self, text="Settings", bg=button_COLOR, width=button_WIDTH, command=lambda: [self.capture.release(),
                                                                                         cv2.destroyAllWindows(),
                                                                                         controller.set_previous_window(
                                                                                             PreviewWindow),
                                                                                         controller.show_frame(SettingsWindow)])
        settings_button.pack(pady=button_pady)

        # Button for continuing to the next page
        continue_button = tk.Button(
            self, text="Continue", bg=button_COLOR, width=button_WIDTH, command=lambda: [self.capture.release(),
                                                                                         cv2.destroyAllWindows(),
                                                                                         controller.show_frame(RecordingWindow)])
        continue_button.pack(pady=button_pady)

    # Show webcam footage for face/eye detection in preview window
    def show_preview_frame(self, cap):
        ret, frame = cap.read()

        # If no frame is returned from cap.read(), return from function since there is no frame data
        if not ret:
            return

        # Converting the recorded image to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Applying filter to remove impurities
        filtered_img = cv2.bilateralFilter(gray_img, 5, 1, 1)

        # Number of faces detected from webcam footage
        faces_detected = self.face_cascade.detectMultiScale(
            filtered_img, 1.3, 5, minSize=(150, 150))

        if len(faces_detected) > 0:

            # Display face detected message on preview frame
            cv2.putText(frame, "Face Detected", (100, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 2)

            # Outline face detection with a rectangle frame
            for (x, y, w, h) in faces_detected:
                frame = cv2.rectangle(
                    frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # roi_gray is the face inputted for eye identification
                roi_gray = gray_img[y:y+h, x:x+w]

                roi_color = frame[y:y+h, x:x+w]

                # Number of eyes detected from webcam footage
                eyes_detected = self.eye_cascade.detectMultiScale(roi_gray)

                for (ex, ey, ew, eh) in eyes_detected:
                    cv2.rectangle(roi_color, (ex, ey),
                                  (ex+ew, ey+eh), (0, 255, 0), 2)

                # Detecting the numbers of eyes (must be greater than 2 to be considered open)
                if len(eyes_detected) >= 2:
                    cv2.putText(frame,
                                "Eyes Detected", (100, 100),
                                cv2.FONT_HERSHEY_PLAIN, 2,
                                (0, 255, 0), 2)
                else:
                    cv2.putText(frame,
                                "You blinked!", (100, 100),
                                cv2.FONT_HERSHEY_PLAIN, 2,
                                (0, 255, 0), 2)
        else:
            cv2.putText(frame,
                        "No Face Detected", (70, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 2)

        # Create a color image
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Create image memory
        image = Image.fromarray(cv2image)

        # Create tkinter image
        imagetk = ImageTk.PhotoImage(image=image)

        # Show image (webcam footage) on preview frame
        self.preview_frame.imagetk = imagetk
        self.preview_frame.configure(image=imagetk)

        # Update preview frame every 25 milliseconds (roughly 40 FPS)
        self.preview_frame.after(25, self.show_preview_frame, cap)

    def start(self):
        self.controller.geometry("{}x{}".format(cap_WIDTH, cap_HEIGHT))
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.show_preview_frame(self.capture)


class RecordingWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        background_image = tk.PhotoImage(file=cap_background_IMAGE)
        BGlabel = tk.Label(self, image=background_image)
        BGlabel.image = background_image
        BGlabel.place(x=0, y=0, width=cap_WIDTH, height=cap_HEIGHT)

        notice_label = tk.Label(
            self, text="Adjust position if no face or eyes are detected", bg=label_COLOR, font=LARGE_FONT)
        notice_label.pack()

        # Initialize face and eye cascade classifiers from cv2 imported xml files
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE)

        # Initialize video capture class variable
        self.capture = None

        # Initialize settings
        self.settings = self.controller.get_settings()

        # Initialize blink and face detection setting
        self.blink_detected = False
        self.face_undetected = False

        # Initalize start timer for blink and face threshold detection
        self.blink_start_time = time.time()
        self.face_start_time = time.time()

        # Frame creation for webcam view to be placed in
        self.recording_frame = tk.Label(self)
        self.recording_frame.pack(pady=10, padx=10)

        # Create DrowsyData object for datapoint storage
        self.data = DrowsyData()

        # Button leading to the Settings page
        settings_button = tk.Button(
            self, text="Settings", bg=button_COLOR, width=button_WIDTH, command=lambda: [self.capture.release(), cv2.destroyAllWindows(),
                                                                                         self.data.save_data_to_file(
                                                                                             data_filename),
                                                                                         self.data.clear_datapoints(),
                                                                                         controller.set_previous_window(
                                                                                             RecordingWindow),
                                                                                         controller.show_frame(SettingsWindow)])
        settings_button.pack(pady=button_pady)

        # Button for ending the recording and moving to Alert/Data Window
        end_button = tk.Button(
            self, text="End Record", bg=button_COLOR, width=button_WIDTH, command=lambda: [self.capture.release(), cv2.destroyAllWindows(),
                                                                                           self.data.save_data_to_file(
                                                                                               data_filename),
                                                                                           self.data.clear_datapoints(),
                                                                                           controller.show_frame(ResultsWindow)])
        end_button.pack(pady=button_pady)

    # Show webcam footage for face/eye detection in preview window
    def show_recording_frame(self, cap):
        ret, frame = cap.read()

        # If no frame is returned from cap.read(), return from function since there is no frame data
        if not ret:
            return

        # Converting the recorded image to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Applying filter to remove impurities
        filtered_img = cv2.bilateralFilter(gray_img, 5, 1, 1)

        # Number of faces detected from webcam footage
        faces_detected = self.face_cascade.detectMultiScale(
            filtered_img, 1.3, 5, minSize=(150, 150))

        if len(faces_detected) > 0:
            # Display Face detected message on recording frame
            cv2.putText(frame, "Face Detected", (100, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

            self.face_undetected = False

            # roi_gray is the face inputted for eye identification
            for (x, y, w, h) in faces_detected:
                roi_gray = gray_img[y:y+h, x:x+w]

            # Number of eyes detected from webcam footage
            eyes_detected = self.eye_cascade.detectMultiScale(roi_gray)

            # Detecting the numbers of eyes (must be greater than 2 to be considered open)
            if len(eyes_detected) >= 2:
                cv2.putText(frame, "Eyes Detected", (100, 100),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                # Check if user was just blinking and now eyes are open, if so then a blink just ended and
                # data needs to be collected (if applicable)
                if self.blink_detected:
                    self.blink_detected = False
                    blink_elapsed_time = float("{0:.2f}".format(
                        time.time() - self.blink_start_time))
                    print("EYES DETECTED- ELAPSED TIME " +
                          str(blink_elapsed_time))

                    # Check if blink is past the Eye closure threshold, if so the blink was long enough
                    # to generate a Type 1 datapoint
                    if blink_elapsed_time >= self.settings["eye_thresh"]:
                        self.data.add_datapoint(
                            str(datetime.now()), blink_elapsed_time, 1)

            else:
                # Display No eyes detected message on recording frame
                cv2.putText(frame, "No Eyes Detected", (100, 100),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                # If a blink has not been detected yet then mark the variable to True and
                # gather the start time of the blink
                if not self.blink_detected:
                    self.blink_detected = True
                    self.blink_start_time = time.time()

                # If a blink has been detected, check if the blink time is greater than the alarm_thresh,
                # if so then a datapoint needs to be saved and alarm window raised
                else:
                    blink_elapsed_time = float("{0:.2f}".format(
                        time.time() - self.blink_start_time))

                    if blink_elapsed_time >= self.settings["alarm_thresh"]:
                        self.data.add_datapoint(
                            str(datetime.now()), blink_elapsed_time, 2)
                        self.blink_detected = False

                        self.capture.release()
                        cv2.destroyAllWindows()

                        self.data.save_data_to_file(data_filename)
                        self.data.clear_datapoints()

                        self.controller.show_frame(AlertWindow)
        else:
            # Display No face detected message on recording frame
            cv2.putText(frame, "No Face Detected", (100, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

            # If face was marked as detected and now isn't visible, mark face as undetected and gather start time
            if not self.face_undetected:
                self.face_undetected = True
                self.face_start_time = time.time()

            # If face is undetected and isn't visible, check if undetected time is greater than the alarm thresh,
            # if so then a datapoint needs to be saved and alarm window raised
            else:
                face_elapsed_time = float("{0:.2f}".format(
                    time.time() - self.face_start_time))
                if face_elapsed_time >= self.settings["alarm_thresh"]:
                    self.data.add_datapoint(
                        str(datetime.now()), face_elapsed_time, 2)
                    self.face_undetected = False

                    self.capture.release()
                    cv2.destroyAllWindows()

                    self.data.save_data_to_file(data_filename)
                    self.data.clear_datapoints()

                    self.controller.show_frame(AlertWindow)

        # Create a color image
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Create image memory
        image = Image.fromarray(cv2image)

        # Create tkinter image
        imagetk = ImageTk.PhotoImage(image=image)

        # Show image (webcam footage) on preview frame
        self.recording_frame.imagetk = imagetk
        self.recording_frame.configure(image=imagetk)

        # Update preview frame every 25 milliseconds (roughly 40 FPS)
        self.recording_frame.after(25, self.show_recording_frame, cap)

    def start(self):
        self.controller.geometry("{}x{}".format(cap_WIDTH, cap_HEIGHT))
        self.settings = self.controller.get_settings()
        self.blink_detected = False
        self.face_undetected = False
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.show_recording_frame(self.capture)


class AlertWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        background_image = tk.PhotoImage(file=background_IMAGE)
        BGlabel = tk.Label(self, image=background_image)
        BGlabel.image = background_image
        BGlabel.place(x=0, y=0, width=nocap_WIDTH, height=nocap_HEIGHT)

        self.minutes = 0
        self.seconds = 0
        self.snoozed = False
        self.wakeup = False

        # Get alarm settings
        self.settings_dict = None
        self.snooze_time = 0

        self.label = tk.Label(
            self, text="Alert!\nYou are falling asleep!", bg=label_COLOR, font=LARGE_FONT, fg="#f00", width=30)
        self.label.pack(pady=10, padx=10)

        wake_button = tk.Button(self, text="Wake Up", bg=button_COLOR, width=button_WIDTH,
                                command=lambda: [self.wake(), self.generate_datapoint(),
                                                 controller.show_frame(RecordingWindow)])
        wake_button.pack(pady=5)

        snooze_button = tk.Button(
            self, text="Snooze",  bg=button_COLOR, width=button_WIDTH, command=lambda: self.snooze())
        snooze_button.pack(pady=5)

    def alert(self):
        if not self.wakeup:
            if not self.snoozed:
                # self.refresh_label()
                winsound.Beep(FREQ, DURATION)

                # Request tkinter to call self.alert after 1s (the delay is given in ms)
                self.after(1000, self.alert)

    def refresh_label(self):
        if self.seconds + self.minutes > 0:
            text = "Snoozed: " + str(int(self.minutes)) + \
                " min " + str(int(self.seconds)) + " sec"
            self.label.configure(text=text)
            if self.seconds == 0:
                self.minutes -= 1
                self.seconds = 59
            else:
                self.seconds -= 1
            self.label.after(1000, self.refresh_label)
        else:
            self.label.configure(text="Alert!\nYou are falling asleep!")
            self.snoozed = False
            self.alert()

    def wake(self):
        self.wakeup = True
        self.snoozed = False
        self.seconds = 0
        self.minutes = 0

    @staticmethod
    def generate_datapoint():
        data = DrowsyData()
        data.add_datapoint(str(datetime.now()), 0, 3)
        data.save_data_to_file(data_filename)
        del data

    def snooze(self):
        if self.snoozed:
            self.minutes = (self.snooze_time / 1000) / 60
            self.seconds = (self.snooze_time / 1000) % 60
        else:
            self.snoozed = True
            self.minutes = (self.snooze_time / 1000) / 60
            self.seconds = (self.snooze_time / 1000) % 60
            self.refresh_label()

    def start(self):
        self.controller.geometry("{}x{}".format(nocap_WIDTH, nocap_HEIGHT))
        self.settings_dict = self.controller.get_settings()
        self.snooze_time = int(self.settings_dict["snooze_thresh"]) * 60 * 1000
        self.wakeup = False
        self.alert()


class SettingsWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        background_image = tk.PhotoImage(file=background_IMAGE)
        BGlabel = tk.Label(self, image=background_image)
        BGlabel.image = background_image
        BGlabel.place(x=0, y=0, width=nocap_WIDTH, height=nocap_HEIGHT)

        # Initialize SettingsWindow member variables
        self.settings_dict = self.controller.get_settings()
        self.ECT_text = tk.IntVar()
        self.alarm_text = tk.IntVar()
        self.snooze_text = tk.IntVar()

        # Top of page label indicating window name
        self.label = tk.Label(self, text="Settings",
                              bg=label_COLOR, font=LARGE_FONT, pady=8)
        self.label.place(anchor='n', x=nocap_WIDTH / 2)

        # Entry boxes and labels for data input
        # Eye closure time
        self.ECT_entry = tk.Entry(
            self, justify='right', textvariable=self.ECT_text, width=10)
        self.ECT_entry.place(anchor='nw', x=20, y=40)

        self.ECT_label = tk.Label(
            self, text="Seconds till detecting closed eyes", bg=label_COLOR, font=SMALL_FONT)
        self.ECT_label.place(anchor='nw', x=90, y=40)

        # Alarm threshold
        self.alarm_threshold_entry = tk.Entry(
            self, justify='right', textvariable=self.alarm_text, width=10)
        self.alarm_threshold_entry.place(anchor='nw', x=20, y=65)

        self.alarm_threshold_label = tk.Label(
            self, text="Seconds till alarm alert", bg=label_COLOR, font=SMALL_FONT)
        self.alarm_threshold_label.place(anchor='nw', x=90, y=65)

        # Snooze threshold
        self.snooze_threshold_entry = tk.Entry(
            self, justify='right', textvariable=self.snooze_text, width=10)
        self.snooze_threshold_entry.place(anchor='nw', x=20, y=90)

        self.snooze_label = tk.Label(
            self, text="Minutes till ending snooze", bg=label_COLOR, font=SMALL_FONT)
        self.snooze_label.place(anchor='nw', x=90, y=90)

        # Bottom of page buttons for navigation
        self.save_button = tk.Button(self, text="Save", width=button_WIDTH, bg=button_COLOR,
                                     command=lambda: [self.save_data(),
                                                      controller.show_frame(controller.get_previous_window())])
        self.save_button.place(anchor='n', x=nocap_WIDTH / 2, y=185)

        self.back_button = tk.Button(self, text="Back", width=button_WIDTH, bg=button_COLOR,
                                     command=lambda: controller.show_frame(controller.get_previous_window()))
        self.back_button.place(anchor='n', x=nocap_WIDTH / 2, y=216)

    def update_fields(self):
        while True:
            try:
                self.ECT_text.set(self.settings_dict["eye_thresh"])
                self.alarm_text.set(self.settings_dict["alarm_thresh"])
                self.snooze_text.set(self.settings_dict["snooze_thresh"])
                break
            except KeyError:
                tk.messagebox.showerror("ERROR",
                                        "File Corruption detected!\n\nReverting 'settings.json' to default values")
                self.controller.fix_corruption()

    def save_data(self):
        saved_data = {}
        try:
            saved_data["eye_thresh"] = int(self.ECT_text.get())
            saved_data["alarm_thresh"] = int(self.alarm_text.get())
            saved_data["snooze_thresh"] = int(self.snooze_text.get())
        except tk.TclError:
            tk.messagebox.showwarning(
                "WARNING", "Only numerals [1-9] allowed in entry fields")
            return
        settings_file = open(settings_filename, "w")
        settings_file.write(json.dumps(saved_data, indent=4))
        tk.messagebox.showinfo("Settings", "Settings saved to disk")

    def start(self):
        self.controller.geometry("{}x{}".format(nocap_WIDTH, nocap_HEIGHT))
        self.settings_dict = self.controller.get_settings()
        self.update_fields()


class ResultsWindow(tk.Frame):
    def __init__(self, parent, controller):
        self.controller = controller

        # Initialize Tkinter frame
        tk.Frame.__init__(self, parent)

        background_image = tk.PhotoImage(file=background_IMAGE)
        BGlabel = tk.Label(self, image=background_image)
        BGlabel.image = background_image
        BGlabel.place(x=0, y=0, width=nocap_WIDTH, height=nocap_HEIGHT)

        scatter_img = PhotoImage(file=scatter_sample)
        scatter_label = Label(self, image=scatter_img)
        scatter_label.image = scatter_img

        histogram_img = PhotoImage(file=histogram_sample)
        histogram_label = Label(self, image=histogram_img)
        histogram_label.image = histogram_img

        pie_img = PhotoImage(file=pie_sample)
        pie_label = Label(self, image=pie_img)
        pie_label.image = pie_img

        table_img = PhotoImage(file=table_sample)
        table_label = Label(self, image=table_img)
        table_label.image = table_img

        scatter_plot_button = tk.Button(self, text="Scatter Plot", image=scatter_img,
                                        command=lambda: [self.generate_scatter_plot()])
        scatter_plot_button.place(anchor='nw', x=40, y=10)

        bar_graph_button = tk.Button(self, text="Bar Graph", image=histogram_img,
                                     command=lambda: [self.generate_bar_graph()])
        bar_graph_button.place(anchor='nw', x=184, y=10)

        pie_chart_button = tk.Button(self, text="Pie Chart", image=pie_img,
                                     command=lambda: [self.generate_pie_chart()])
        pie_chart_button.place(anchor='nw', x=40, y=100)

        table_button = tk.Button(self, text="Table", image=table_img,
                                 command=lambda: [self.generate_sleep_table()])
        table_button.place(anchor='nw', x=184, y=100)

        menu_button = tk.Button(self, text="Main Menu", bg=button_COLOR, width=button_WIDTH,
                                command=lambda: [controller.show_frame(MainWindow)])
        menu_button.place(anchor='nw', x=25, y=200)

        self.export_filename = None

        export_button = tk.Button(self, text="Export Data", bg=button_COLOR, width=button_WIDTH,
                                  command=lambda: [self.set_export_name(),
                                                   self.export_data(data_filename,
                                                                    self.export_filename,
                                                                    export_folder_name)])
        export_button.place(anchor='nw', x=125, y=200)

        end_button = tk.Button(
            self, text="End Program", bg=button_COLOR, width=button_WIDTH, command=self.quit)
        end_button.place(anchor='nw', x=225, y=200)

    def set_export_name(self):
        self.export_filename = str(datetime.now()).replace(
            ' ', '_').replace(':', '_')[:19] + '_data.csv'

    def generate_scatter_plot(self):
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        dataframe = pd.read_csv(
            'data.csv', encoding="utf-8", error_bad_lines=False)

        if not dataframe.empty:
            dataframe['time'] = pd.to_datetime(dataframe['time'])
            plt.figure(figsize=(7, 7))
            dataframe.sort_values('time', inplace=True)
            plt.plot_date(dataframe['time'], dataframe['duration'])
            plt.gcf().autofmt_xdate()
            time_format = mdates.DateFormatter('%H:%M:%S')
            plt.gca().xaxis.set_major_formatter(time_format)
            plt.tight_layout()
            plt.gcf().subplots_adjust(left=0.08, top=0.95, bottom=0.12)
            plt.title('Time vs Blink Duration')
            plt.xlabel('Time (HH:MM:SS)')
            plt.ylabel('Blink Duration (sec)')
            plt.show()
        else:
            label = tk.Label(self, text="Not Enough Data",
                             font=LARGE_FONT, bg=button_COLOR)
            label.pack(pady=10)
            label.after(1000, lambda: label.destroy())

    def generate_bar_graph(self):
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        dataframe = pd.read_csv(
            'data.csv', encoding="utf-8", error_bad_lines=False)

        if not dataframe.empty:
            dataframe['time'] = pd.to_datetime(dataframe['time'])
            plt.figure(figsize=(7, 7))
            ax = (dataframe.groupby(dataframe["time"].dt.hour)[
                  'duration'].nunique()).plot(kind="bar")
            ax.set_title("Detected Blinks by Hour")
            ax.set_xlabel("Hour of the Day (hour)")
            ax.set_ylabel("Number of Threshold Blinks")
            plt.show()
        else:
            label = tk.Label(self, text="Not Enough Data",
                             font=LARGE_FONT, bg=button_COLOR)
            label.pack()
            label.after(1000, lambda: label.destroy())

    def generate_pie_chart(self):
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        dataframe = pd.read_csv(
            'data.csv', encoding="utf-8", error_bad_lines=False)

        if not dataframe.empty:
            dataframe['duration'] = dataframe['duration'].astype(int)

            unique_vals = dataframe['duration'].unique()
            labels = np.delete(unique_vals, np.where(unique_vals == 0), None)
            label_counts = []
            for label in labels:
                label_counts.append(
                    len(dataframe[dataframe['duration'] == label]))

            explode = []
            for label in labels:
                explode.append(0.1)
            piechart = plt.figure(figsize=(7, 7))
            piechart.suptitle("Blink Duration Frequency (sec)")

            explode = []
            for label in labels:
                explode.append(0.1)

            plt.pie(label_counts, labels=labels, shadow=True,
                    explode=explode, autopct='%1.1f%%')
            plt.axis('equal')
            plt.show()
        else:
            label = tk.Label(self, text="Not Enough Data",
                             font=LARGE_FONT, bg=button_COLOR)
            label.pack()
            label.after(1000, lambda: label.destroy())

    def generate_sleep_table(self):
        raw_data = DrowsyData().get_data_from_file('data.csv')

        sleep_data = {}
        for key in raw_data.keys():
            if raw_data[key][1] in [2, 3]:
                sleep_data.update({key: raw_data[key]})

        if len(sleep_data) == 0:
            label = tk.Label(self, text="Not Enough Data",
                             font=LARGE_FONT, bg=button_COLOR)
            label.pack()
            label.after(1000, lambda: label.destroy())
            return

        table_data = []
        iter_dict = iter(sleep_data)
        for key in iter_dict:
            if sleep_data[key][1] == 2:
                time_format = '%Y-%m-%d %H:%M:%S.%f'
                sleep_start = datetime.strptime(
                    key, time_format) - timedelta(seconds=int(float(sleep_data[key][0])))
                sleep_end = datetime.strptime(next(iter_dict), time_format)
                sleep_dur = sleep_end - sleep_start
                table_data.append([sleep_start.strftime(time_format)[:21],
                                   sleep_end.strftime(time_format)[:21],
                                   str(sleep_dur)[:9]])

        # Calculate figure's height to either be a small table or big table
        if len(table_data) > 12:
            height = 9
        else:
            height = 4

        plt.style.use('seaborn')
        figure, ax = plt.subplots(figsize=(7, height))
        table = ax.table(cellText=table_data,
                         colLabels=['Sleep Start Time',
                                    'Sleep End time', 'Duration of Sleep'],
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.25, 1.25)
        ax.axis('off')

        plt.show()

    @staticmethod
    def export_data(cur_filename, new_filename, export_dir_name):
        # Create a new directory using the "export_dir_name" var as the name
        try:
            os.mkdir(export_dir_name)
        except FileExistsError:
            pass

        # Get original working directory that the program normally uses
        orig_directory = os.getcwd()

        # Change directory to the export directory
        os.chdir(export_dir_name)

        # Copy file from original directory to new directory
        shutil.copy(orig_directory + '\\' + cur_filename, new_filename)

        # Change directory to the original working directory
        os.chdir(orig_directory)

    def start(self):
        self.controller.geometry("{}x{}".format(nocap_WIDTH, nocap_HEIGHT))
