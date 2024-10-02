# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from tensorflow.keras import layers, models
# from tensorflow.keras.models import load_model  # Uvezi funkciju za učitavanje modela
#
# # Putanja do podataka
# base_dir = 'images/'
# image_list = os.listdir(base_dir)
#
# # Podela na train i test
# np.random.shuffle(image_list)
# train_size = int(0.8 * len(image_list))
# train_images = image_list[:train_size]
# test_images = image_list[train_size:]
#
# # Priprema podataka sa oznakama
# def get_labels(image_list):
#     labels = []
#     for img in image_list:
#         # Pretpostavljam da ime slike sadrži klasu
#         label = img.split('_')[0]  # Primer: "labrador_01.jpg" -> "labrador"
#         labels.append(label)
#     return labels
#
# train_labels = get_labels(train_images)
# test_labels = get_labels(test_images)
#
# # Kreiranje generatora podataka
# train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
# test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
#
# train_generator = train_datagen.flow_from_dataframe(
#     dataframe=pd.DataFrame({'filename': train_images, 'label': train_labels}),
#     directory=base_dir,
#     x_col='filename',
#     y_col='label',
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=True
# )
#
# test_generator = test_datagen.flow_from_dataframe(
#     dataframe=pd.DataFrame({'filename': test_images, 'label': test_labels}),
#     directory=base_dir,
#     x_col='filename',
#     y_col='label',
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=False
# )
#
# # Proverite da li model već postoji
# model_path = 'dog_classifier_model.h5'
# if os.path.exists(model_path):
#     model = load_model(model_path)  # Učitavanje sačuvanog modela
# else:
#     # Kreiranje modela ako model ne postoji
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(128, (3, 3), activation='relu'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(len(train_generator.class_indices), activation='softmax')
#     ])
#
#     # Kompajliranje modela
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     # Obučavanje modela
#     history = model.fit(
#         train_generator,
#         steps_per_epoch=len(train_generator),
#         epochs=10,
#         validation_data=test_generator,
#         validation_steps=len(test_generator)
#     )
#
#     # Čuvanje modela
#     model.save(model_path)
#
# # Evaluacija modela
# test_loss, test_acc = model.evaluate(test_generator, verbose=2)
# print(f'\nTest accuracy: {test_acc:.2f}')
#
# # Prikaz grafika tačnosti
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# plt.show()
#
# # Funkcija za predikciju nove slike
# def predict_image(model, image_path):
#     img = load_img(image_path, target_size=(150, 150))  # Učitavanje slike
#     img_array = img_to_array(img) / 255.0  # Normalizacija
#     img_array = np.expand_dims(img_array, axis=0)  # Dodavanje dimenzije
#
#     predictions = model.predict(img_array)  # Predikcija
#     class_idx = np.argmax(predictions[0])  # Indeks klase sa najvišim skorom
#     class_label = list(train_generator.class_indices.keys())[class_idx]  # Klasa
#
#     return class_label
#
# # Korišćenje funkcije za predikciju
# image_path = input("Unesite putanju do slike za klasifikaciju: ")
# predicted_class = predict_image(model, image_path)
# print(f'Predikovana klasa: {predicted_class}')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models

# Putanja do podataka
base_dir = 'images/'
image_list = os.listdir(base_dir)

# Podela na train i test
np.random.shuffle(image_list)  # Nasumično mešanje slika
train_size = int(0.8 * len(image_list))  # 80% za obučavanje
train_images = image_list[:train_size]
test_images = image_list[train_size:]

# Priprema podataka sa oznakama
def get_labels(image_list):
    labels = []
    for img in image_list:
        label = img.split('_')[0]  # Pretpostavka da ime slike sadrži klasu
        labels.append(label)
    return labels

train_labels = get_labels(train_images)
test_labels = get_labels(test_images)

# Kreiranje generatora podataka
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_images, 'label': train_labels}),
    directory=base_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_images, 'label': test_labels}),
    directory=base_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Učitajte prethodno sačuvani model
model = models.load_model('dog_classifier_model.h5')  # Zamenite sa pravom putanjom do modela

# Učitavanje slike
image_path = input("Unesite putanju do slike: ")

# Preprocesiranje slike
img = load_img(image_path, target_size=(150, 150))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Dodajte dimenziju za batch

# Normalizacija
img_array /= 255.0

# Predikcija
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])  # Dobijte klasu sa najvećom verovatnoćom

# Prikaz rezultata
class_labels = list(train_generator.class_indices.keys())  # Dobijte imena klasa
print(f'Predikcija: {class_labels[predicted_class]}')
