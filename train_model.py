import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
DATA_DIR = 'Sims_Dataset_Clean'
SEED = 42
NUM_CLASSES = 6
class_names = ['Angry', 'Embarrassed', 'Happy', 'Sad', 'Tense', 'Uncomfortable']

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    label_mode='categorical',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.3,
    subset="training",
    class_names=class_names 
)

temp_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    label_mode='categorical',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.3,
    subset="validation",
    class_names=class_names
)

val_batches = len(temp_dataset) // 2
val_ds = temp_dataset.take(val_batches)
test_ds = temp_dataset.skip(val_batches)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False 

# Model
model = models.Sequential()
model.add(layers.Input(shape=(160, 160, 3)))

model.add(layers.RandomFlip("horizontal"))
model.add(layers.RandomRotation(0.05))
model.add(layers.RandomZoom(0.05))

model.add(layers.Rescaling(1./127.5, offset=-1))
model.add(base_model)

model.add(layers.GlobalAveragePooling2D())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=val_ds, epochs=25)

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    batch_true = np.argmax(labels.numpy(), axis=1)
    batch_pred = np.argmax(preds, axis=1)
    
    for true_val in batch_true:
        y_true.append(true_val)
    
    for pred_val in batch_pred:
        y_pred.append(pred_val)

# Results
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Sims 4 Emotion Detection - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted')
#plt.show()