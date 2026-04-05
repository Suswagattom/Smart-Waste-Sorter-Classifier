This project uses a Deep Learning AI model to detect and classify 10 different types of garbage in real-time, mapping them into broad waste disposal categories (Recyclable, Dry, Wet, Biohazardous). It features a live camera feed with a built-in confidence lock-on and voice announcements using text-to-speech.

Model Details
Base Architecture: Xception (Transfer Learning via TensorFlow/Keras)

Training Method: 5-Fold Cross-Validation for robust, generalized accuracy.

Classes (10): Battery, Biological, Cardboard, Clothes, Glass, Metal, Paper, Plastic, Shoes, Trash.

Image Dimensions: 224x224

Dataset: Trained on thousands of categorized images. Special thanks to Kaggle: Garbage Classification V2 for providing the core dataset.

Setup & Configuration
Before running the scripts, you must configure the file paths in the code to point to your local directories.

1. prepare_data.py (Dataset Organizer)

Line 3: Update source_dir = r"YOUR_RAW_DATASET_PATH"

Line 4: Update dest_dir = r"YOUR_NEW_SORTED_DATASET_PATH"

2. kfold_training.py (Model Training)

Line 9: Update data_dir = r"YOUR_TRAINING_DATASET_PATH"

3. live_detection.py (Live Camera Feed)

Line 8: Update model = tf.keras.models.load_model(r'YOUR_MODEL_PATH') (Point this to your saved best_kfold_model.h5 file).

Line 28: Update video_url = "YOUR_CAMERA_FEED" (Set this to 0 if you are using your default computer webcam, or paste an IP camera URL).

Acknowledgements
Dataset provided by sumn2u on Kaggle.

Portions of this project's code structure and logic were developed with the assistance of AI.