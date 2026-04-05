import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import time

print("Loading 10-Class K-Fold AI Model...")
model = tf.keras.models.load_model(r'')    #Model location

class_names = ['Battery', 'Biological', 'Cardboard', 'Clothes', 'Glass', 'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash']

waste_map = {
    'Battery': 'biohazardous',
    'Biological': 'wet',
    'Cardboard': 'dry',
    'Clothes': 'dry',
    'Glass': 'recyclable',
    'Metal': 'recyclable',
    'Paper': 'dry',
    'Plastic': 'recyclable',
    'Shoes': 'dry',
    'Trash': 'dry'
}

def speak_message(message):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) 
    engine.say(message)
    engine.runAndWait()

video_url = ""    #Video url
cap = cv2.VideoCapture(video_url)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 

required_hold_time = 4.0  
timeout_limit = 10.0    
normalize_limit = 10.0  

current_analyzing_class = ""
start_analysis_time = 0
first_detected_time = 0 

is_locked = False
has_failed = False
locked_time = 0
frame_count = 0

display_text = "Waiting for trash..."
box_color = (255, 255, 0) 

while True:
    ret, frame = cap.read()
    
    if not ret: 
        continue 

    h, w, _ = frame.shape
    box_size = 300
    x1 = int((w / 2) - (box_size / 2))
    y1 = int((h / 2) - (box_size / 2))
    x2 = int((w / 2) + (box_size / 2))
    y2 = int((h / 2) + (box_size / 2))

    frame_count += 1

    if frame_count % 5 == 0:
        roi = frame[y1:y2, x1:x2]
        img = cv2.resize(roi, (224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array, verbose=0)
        score = predictions[0] 
        
        predicted_item = class_names[np.argmax(score)]
        broad_category = waste_map[predicted_item]
        confidence = 100 * np.max(score)

        if confidence > 60.0:
            if first_detected_time == 0:
                first_detected_time = time.time()
                
            time_since_trash_landed = time.time() - first_detected_time

            if is_locked:
                if (time.time() - locked_time) > normalize_limit:
                    display_text = "Normalized. Please clear box."
                    box_color = (150, 150, 150) 
            
            elif has_failed:
                display_text = "FAILED. Please remove item."
                box_color = (0, 0, 255) 
            
            else:
                if time_since_trash_landed > timeout_limit:
                    has_failed = True
                    display_text = "FAILED TO EVALUATE"
                    box_color = (0, 0, 255) 
                    threading.Thread(target=speak_message, args=("Failed to evaluate. Sorry.",), daemon=True).start()
                
                else:
                    if predicted_item == current_analyzing_class:
                        time_elapsed = time.time() - start_analysis_time
                        progress_percent = min(int((time_elapsed / required_hold_time) * 100), 99)
                        
                        display_text = f"Analyzing {predicted_item}: {progress_percent}%"
                        box_color = (0, 165, 255) 
                        
                        if time_elapsed >= required_hold_time:
                            is_locked = True
                            locked_time = time.time()
                            display_text = f"LOCKED: {predicted_item} ({broad_category})"
                            
                            if broad_category == 'biohazardous': box_color = (0, 0, 255)
                            elif broad_category == 'wet': box_color = (255, 0, 0)
                            else: box_color = (0, 255, 0)
                            
                            msg = f"This is {predicted_item}, {broad_category} waste"
                            threading.Thread(target=speak_message, args=(msg,), daemon=True).start()
                    else:
                        current_analyzing_class = predicted_item
                        start_analysis_time = time.time()
                        display_text = f"Analyzing {predicted_item}..."
                        box_color = (0, 165, 255)
        else:
            is_locked = False
            has_failed = False
            first_detected_time = 0
            current_analyzing_class = ""
            display_text = "Waiting for trash..."
            box_color = (255, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2 if not is_locked else 4)
    cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 3)
    cv2.imshow('Smart Waste Sorter - Live Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()