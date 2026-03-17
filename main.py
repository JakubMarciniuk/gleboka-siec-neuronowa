import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ==========================================
# 1. KONFIGURACJA
# ==========================================
TEST_PATH = r"C:\Users\jakub\Downloads\archive\DATASET\TEST"
TRAIN_PATH = r"C:\Users\jakub\Downloads\archive\DATASET\TRAIN"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 30

# ==========================================
# 2. PRZYGOTOWANIE DANYCH
# ==========================================
print("Wczytywanie danych...")

# Normalizacja
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'  # Model uczy się na szarościach
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono folderów! Sprawdź ścieżki.")
    exit()

# ==========================================
# 3. BUDOWA MODELU
# ==========================================
model = Sequential([
    InputLayer(input_shape=(64, 64, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu', name='hidden_layer'),
    Dense(1, activation='sigmoid', name='output_layer')
])

optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy', 'mse'])

# ==========================================
# 4. CALLBACKI
# ==========================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)


class WeightsHistory(Callback):
    def on_train_begin(self, logs={}):
        self.hidden_weights = []
        self.output_weights = []

    def on_epoch_end(self, epoch, logs={}):
        h_w = self.model.get_layer('hidden_layer').get_weights()[0]
        o_w = self.model.get_layer('output_layer').get_weights()[0]
        self.hidden_weights.append(h_w.flatten())
        self.output_weights.append(o_w.flatten())


weights_history = WeightsHistory()

# ==========================================
# 5. TRENING
# ==========================================
print("\nRozpoczynam trening...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stop, reduce_lr, weights_history]
)

# ==========================================
# 6. GENEROWANIE WYKRESÓW
# ==========================================
print("\nGenerowanie wykresów...")

# --- Wykres MSE ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='MSE (Train)')
plt.plot(history.history['val_loss'], label='MSE (Validation)')
plt.title('Błąd średniokwadratowy (MSE)')
plt.xlabel('Epoka')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.savefig('wykres_mse.png')
plt.close()

# --- Wykres Błędu ---
train_err = 1 - np.array(history.history['accuracy'])
val_err = 1 - np.array(history.history['val_accuracy'])
plt.figure(figsize=(10, 5))
plt.plot(train_err, label='Błąd klasyfikacji (Train)')
plt.plot(val_err, label='Błąd klasyfikacji (Validation)')
plt.title('Błąd klasyfikacji (Próg 0.5)')
plt.xlabel('Epoka')
plt.ylabel('Poziom błędu')
plt.legend()
plt.grid(True)
plt.savefig('wykres_bledow.png')
plt.close()

# --- Wykres Wag ---
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(weights_history.hidden_weights[0], bins=50, color='blue', alpha=0.7)
plt.title('Wagi warstwy ukrytej (Początek)')
plt.subplot(2, 2, 2)
plt.hist(weights_history.hidden_weights[-1], bins=50, color='green', alpha=0.7)
plt.title('Wagi warstwy ukrytej (Koniec)')
plt.subplot(2, 2, 3)
plt.hist(weights_history.output_weights[0], bins=50, color='blue', alpha=0.7)
plt.title('Wagi warstwy wyjściowej (Początek)')
plt.subplot(2, 2, 4)
plt.hist(weights_history.output_weights[-1], bins=50, color='green', alpha=0.7)
plt.title('Wagi warstwy wyjściowej (Koniec)')
plt.tight_layout()
plt.savefig('wykres_wag.png')
plt.close()


# ==========================================
# 7. PRZYKŁAD KLASYFIKACJI (POPRAWIONY)
# ==========================================
def visualize_prediction(model, base_path, class_indices):
    try:
        # Losowanie
        available_classes = os.listdir(base_path)
        if not available_classes: return
        random_class = np.random.choice(available_classes)
        class_path = os.path.join(base_path, random_class)

        images = os.listdir(class_path)
        if not images: return
        random_image_name = np.random.choice(images)
        image_path = os.path.join(class_path, random_image_name)

        # --- KROK 1: Wczytanie dla MODELU (Grayscale) ---
        img_model = load_img(
            image_path, target_size=IMG_SIZE, color_mode='grayscale'
        )
        img_array = img_to_array(img_model)
        img_array_norm = np.expand_dims(img_array, axis=0) / 255.0

        # --- KROK 2: Wczytanie dla LUDZKIEGO OKA (RGB - Kolor) ---
        img_display = load_img(
            image_path, target_size=IMG_SIZE, color_mode='rgb'
        )

        # Predykcja (na podstawie wersji szarej!)
        prediction = model.predict(img_array_norm)[0][0]

        # Dekodowanie wyniku
        idx_to_label = {v: k for k, v in class_indices.items()}
        if prediction > 0.5:
            pred_label = idx_to_label[1]
            confidence = prediction
        else:
            pred_label = idx_to_label[0]
            confidence = 1 - prediction
        real_label = random_class

        # Rysowanie (wersji kolorowej!)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_display)

        color = 'green' if pred_label == real_label else 'red'
        title_text = (f"Prawdziwa klasa: {real_label}\n"
                      f"Predykcja: {pred_label} ({confidence * 100:.2f}%)")
        plt.title(title_text, color=color, fontsize=14)
        plt.axis('off')

        plt.savefig('przyklad_klasyfikacji.png')
        print("Zapisano: przyklad_klasyfikacji.png (w kolorze)")

    except Exception as e:
        print(f"Błąd podczas generowania przykładu: {e}")


visualize_prediction(model, TEST_PATH, train_generator.class_indices)

print("\nGotowe! Wygenerowano 4 pliki .png do sprawozdania.")