Sign Language Recognition Web Application
Overview
This project is a Flask-based web application for real-time sign language recognition using a pre-trained TensorFlow model. The application uses a webcam feed to detect hand signs and classify them into predefined labels.

Features

Real-time sign language recognition using a webcam.
Displays predictions and confidence levels on the video feed.
Mobile-responsive interface using Bootstrap.
Requirements

Python 3.x
Flask
OpenCV
NumPy
TensorFlow and Keras
CVZone
Code Structure

app.py: The main Flask application file handling routes and video streaming.
requirements.txt: List of Python dependencies.
SLR.html: HTML template for the user interface.
static/: (Optional) Folder for static assets like CSS, JavaScript, or images.

Troubleshooting
Video Not Displaying: Ensure your webcam is correctly connected and accessible.
Model Errors: Verify that fmodel.h5 and flabels.txt are properly formatted and located in the project directory.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.


Contact
For questions or feedback, please contact suhaspisupati2004@gmail.com.
