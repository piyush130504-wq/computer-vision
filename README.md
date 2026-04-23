MNIST is the foundational dataset for handwritten digit recognition in machine learning, containing 70,000 grayscale images of digits 0-9.
Dataset Details
	•	Size: 60,000 training images, 10,000 testing images
	•	Format: 28×28 pixel grayscale (784 features per image, values 0-255)
	•	Classes: 10 digits, roughly balanced (~6,000 train samples per class)
	•	Source: Derived from NIST Special Databases 1 & 3 (Census workers + high school students)

What it is



This project usually predicts digits 0–9 from handwritten input, using a model trained on the MNIST dataset and deployed through a simple Streamlit interface. Typical user interactions are either drawing on a canvas or uploading an image, after which the app preprocesses the image and returns the predicted digit and, optionally, confidence scores.
Typical pipeline
A standard computer-vision pipeline for this app is:

	•	Capture input from a drawable canvas or file uploader.
  
	•	Convert the image to grayscale, resize it to 28×28, and normalize pixel values.
  
	•	Feed the processed image to a trained CNN model for classification.
  
	•	Display the predicted digit inside the Streamlit UI.
  
