import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
​
# Diretórios do dataset
train_dir = 'cats_and_dogs/train'
validation_dir = 'cats_and_dogs/validation'
​
# Parâmetros do modelo
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
​
# Geradores de dados com aumento de dados para prevenir overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
​
validation_datagen = ImageDataGenerator(rescale=1./255)
​
# Geradores de imagens
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
​
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
​
# Definição do modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
​
# Compilação do modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
​
# Definição do caminho do modelo
checkpoint_path = "best_model.keras"
​
callbacks = [
    ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]
​
# Treinamento do modelo
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=callbacks
)
​
# Plotagem das curvas de perda e acurácia
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))
​
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia Treino')
plt.plot(epochs_range, val_acc, label='Acurácia Validação')
plt.legend(loc='lower right')
plt.title('Acurácia')
​
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda Treino')
plt.plot(epochs_range, val_loss, label='Perda Validação')
plt.legend(loc='upper right')
plt.title('Perda')
plt.show()
​
# Avaliação no conjunto de teste
test_generator = validation_datagen.flow_from_directory(
    'cats_and_dogs/validation',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)
​
# Carregar o melhor modelo
model.load_weights(checkpoint_path)
​
# Previsão no conjunto de teste
test_labels = test_generator.classes
predictions = (model.predict(test_generator) > 0.5).astype("int32").flatten()
​
# Matriz de confusão
cm = confusion_matrix(test_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
disp.plot()
plt.show()