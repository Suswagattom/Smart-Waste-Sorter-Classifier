import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

data_dir = r""   #Dataset Location
batch_size = 32
img_height = 224
img_width = 224
k_folds = 5 
epochs_per_fold = 10 

print("Scanning folders and building the master dataset...")

filepaths = []
labels = []
class_names = sorted(os.listdir(data_dir))

for label in class_names:
    folder_path = os.path.join(data_dir, label)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, img_file))
            labels.append(label)

df = pd.DataFrame({'filepath': filepaths, 'label': labels})
print(f"Total images found: {len(df)}")
print(f"Classes: {class_names}")

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
  layers.RandomRotation(0.3),
  layers.RandomZoom(0.2),
])

def build_fresh_model():
    base_model = tf.keras.applications.Xception(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False 

    model = models.Sequential([
        data_augmentation, 
        layers.Rescaling(1./127.5, offset=-1), 
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3), 
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_no = 1
fold_accuracies = []

for train_index, val_index in kf.split(df):
    print(f"\n=====================================")
    print(f"   STARTING FOLD {fold_no} OF {k_folds}")
    print(f"=====================================")
    

    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    
    train_ds = train_datagen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size, class_mode='sparse', shuffle=True
    )
    
    val_ds = val_datagen.flow_from_dataframe(
        val_df, x_col='filepath', y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size, class_mode='sparse', shuffle=False
    )
    
    model = build_fresh_model()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_per_fold
    )
    
    scores = model.evaluate(val_ds, verbose=0)
    print(f"\nScore for Fold {fold_no}: Accuracy = {scores[1]*100:.2f}%")
    fold_accuracies.append(scores[1] * 100)
    
    if scores[1] * 100 == max(fold_accuracies):
        model.save('best_kfold_model.h5')
        print("--> New best model saved!")

    fold_no += 1

print("\n=====================================")
print("        K-FOLD TEST COMPLETE")
print("=====================================")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i+1} Accuracy: {acc:.2f}%")

print(f"\n>>> FINAL TRUE ACCURACY: {np.mean(fold_accuracies):.2f}% (+/- {np.std(fold_accuracies):.2f}%) <<<")
print("The best performing model was saved as 'best_kfold_model.h5'")