Executable: https://drive.google.com/file/d/1LWwjalaidxcK_DKf6G6I47KXj67kkBnR/vie

# Drowsiness Detector

### Contributors:
    * Sonali Patil -- s_patil1@u.pacific.edu
    * Jacob Angulo -- j_angulo1@u.pacific.edu
    * Richard Shin -- r_shin2@u.pacific.edu

### Description
The Drowsiness Detector is a webcam enabled software-system that can detect a user's drowsiness
and report time specific data to the user at the end of recording. Drowsiness is determined via
the detection of eye-lip closures that exceed a user-defined threshold. Upon closure of the eyes
a timer will run until the eyes close again. If an eye-closure that meets the aforementioned
criteria is detected an alert will be sent to the user. In addition, a datapoint will be created that stores
the duration of the event and the time the event occured. Datapoints from the session will be stored
in a historical file (CSV type) and datapoints from that particular session will be displayed to
the user. Data graphing will be the method of data communication and the user has the ability to
select what kind of graph(s) they want to see.

### Responsibilities
- Developed blink detection feature for monitoring eye closures that exceeded a user configured threshold to recognize, alert, and log user drowsiness
- Implemented analytics and visualization feature that used data collected during user monitoring session to generate various graphs and reports and provide statistics on drowsiness
- Created GUI windows and front-end components using Tkinter through multiple design reviews and UI validations

### Components
##### Hardware
1. A Windows 10 compatible 1080P Webcam

##### Software
1. Tkinter Library
    * Version 3.9.1
    * A Python GUI package used to create new frameworks and GUI elements
    * Will allow creation of the GUI for the user to interact with
    * Buttons on these GUI pages will be able to make changes to the settings file,
    data points, and alerting system
      
2. Matplotlib Library
    * Version 3.4.1
    * A graping library for Python used to create data visualization models
    * Will be used for the end-of-session data graphs that get
    displayed to the user at the end of the session.
    
3. Winsound module
    * A python module that provides access to the basic sound-playing machinery provided by Windows platforms.
    * Primarily used in the alertWindow class.
   
4. Pillow Library
    * Version 8.2.0
    * The Python Imaging Library adds image processing capabilities to your Python interpreter.
    * We leverage PIL (pillow) in the preview window webcam-view.

5. opencv-python Library
    * Version 4.5.1.48
    * OpenCV is an open source software library for computer vision and machine learning.
    * We leverage opencv-python in the preview window webcam-view.
    
6. NumPy Library
    * Version 1.20.1
    * Numpy (Numerical Python) is a library for scientific computing.
    * A dependency of opencv-python (not used directly in our code).

7. Pandas Library
    * Version 1.2.4
    * Provides data structures that are designed to make working with structured data simple and efficient.
    * We leverage Pandas when creating dataframes from our collected data.
    
8. Specific Window Classes
    * mainWindow | previewWindow | settingsWindow | recordingWindow | alertWindow | resultsWindow
    * These are windows that leverage Tkinter and contain buttons and fields that may affect
    the settings file, alerting system, graphing, and UI flow
      
9. drowsyData Class 
    * Aims to handle interaction with the historical datafile.
    * An array of datapoints collected throughout the session are contained 
      in a drowsyData object
    * The object is session specific, but even when a session ends, all data should be 
    stored in a historical data file
    
10. Eye Detection
    * Uses haarcascade_frontalface_default.xml and haarcascade_eye_tree_eyeglasses.xml files from OpenCV as pretrained models to detect eyes and face in an image.

11. DateTime 
    * A python module that allows for creation of datetime objects
    * We use it as a method for collecting the current time using the date/time as data to be stored
      
      
### Special Notes
* The settings page (leveraging the settingsWindow class) will save the settings into a settings
  file located in the same folder as the program. This file will hold data such as the time threshold for blink detection, time threshold for an alert to be triggered, and the snooze interval.When the drowsiness detection system is instantiated, various calls will be made to retrieve and implementthe values from the settings file. 

### Handled by Executable
This section denotes libraries and packages that need to be "pip intalled" and thus need special care
during the final phase of our project (should be prebuilt or have installation handled for user).
* opencv-python
* numpy
* pandas
* PIL
* matplotlib
