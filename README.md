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
Setup
Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
Install Dependencies
You can install the required Python packages using pip. It's recommended to use a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
Required Files
fmodel.h5: The pre-trained TensorFlow model file.
flabels.txt: The file containing the labels corresponding to the model's output classes.
SLR.html: The HTML template for the web interface.
Prepare Model and Labels
Ensure fmodel.h5 and flabels.txt are placed in the project directory.

Running the Application
Start the Flask Application

Run the following command in your project directory:

bash
Copy code
python app.py
Access the Web Interface

Open your web browser and navigate to http://127.0.0.1:5000 to view the application.

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
