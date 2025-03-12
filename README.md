# Face-Recognition
A Python-based face recognition application using OpenCV. This system detects faces, trains a recognizer, and identifies individuals in real-time.

# Face Recognition System

A comprehensive Python application for face detection and recognition using OpenCV. This system can detect faces, train a recognizer, and identify individuals based on the trained data.

## 📋 Features

- **Face Detection**: Detects faces in real-time using OpenCV's Haar Cascade classifiers
- **Face Recognition**: Identifies detected faces using LBPH Face Recognizer
- **Database Integration**: Stores user information in SQLite database
- **User Management**: Add, delete, and update user entries
- **Training Module**: Train the system with new faces
- **All-in-One Solution**: Complete face recognition pipeline in a single class

## 🗂️ Project Structure

- **main.py**: Entry point for the application, handles the UI and user interaction
- **detector.py**: Contains the face detection logic using Haar Cascades
- **trainer.py**: Trains the LBPH Face Recognizer with images from the dataset
- **FaceRecognition.py**: All-in-one class that encapsulates the entire face recognition process
- **DeleteEntry.py**: Utility to remove entries from the database
- **temp.py**: Contains temporary/utility functions
- **dataset/**: Directory containing face images used for training
- **recognizer/**: Directory storing the trained model files

## 🔧 Requirements

- Python 3.6+
- OpenCV 4.x
- NumPy
- SQLite3
- PIL (Pillow)

## 📦 Installation

1. Clone this repository:
https://github.com/AkulaAnshul/Face-Recognition.git

2. Install required packages:
pip install opencv-python numpy pillow

## 🚀 Usage

### Running the Application

python main.py

### Adding a New Face

1. Run the application
2. Enter the person's ID and name
3. The system will capture multiple images of the face
4. Train the recognizer with the new data

### Training the Recognizer

python trainer.py

## 💻 Code Examples

### Face Detection


python detector.py

## 💻 Code Examples

### Face Detection


## 📊 Database Structure

The SQLite database (`database.db`) contains the following table:

- **Users**: Stores information about recognized individuals
  - id: Unique identifier
  - name: Person's name
  - date_added: When the entry was created

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📞 Contact

If you have any questions or suggestions, please open an issue or contact [akulaanshulrao@gmail.com].

