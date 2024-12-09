import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image, ImageDraw, ImageFont
import shutil
import random
from tensorflow.keras.callbacks import EarlyStopping

# Ścieżka do folderu z danymi
base_dir = r'C:\Python\SN'
dog_dir = os.path.join(base_dir, 'Dog')
cat_dir = os.path.join(base_dir, 'Cat')

# Tworzenie folderów na podział danych i błędy
split_dir = os.path.join(base_dir, 'Podział danych')
validation_errors_dir = os.path.join(base_dir, 'Validation_Errors')
os.makedirs(split_dir, exist_ok=True)
os.makedirs(validation_errors_dir, exist_ok=True)
train_dir = os.path.join(split_dir, 'train')
validation_dir = os.path.join(split_dir, 'validation')
test_dir = os.path.join(split_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Tworzenie folderów dla dodatkowych wyników
conf_matrix_dir = os.path.join(base_dir, 'Confusion_Matrices')
metrics_plot_dir = os.path.join(base_dir, 'Metrics_Plots')
csv_dir = os.path.join(base_dir, 'Metrics_CSV')
image_grid_dir = os.path.join(base_dir, 'Image_Grids')  # Nowy folder na siatki obrazów

os.makedirs(conf_matrix_dir, exist_ok=True)
os.makedirs(metrics_plot_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(image_grid_dir, exist_ok=True)

# Włączenie dynamicznej alokacji pamięci GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Funkcja do walidacji obrazów i logowania błędów
def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except (IOError, SyntaxError) as e:
        log_file = os.path.join(base_dir, 'log.txt')
        with open(log_file, 'a') as log:
            log.write(f"Invalid image file: {file_path}, Error: {e}\n")
        error_dest = os.path.join(validation_errors_dir, os.path.basename(file_path))
        shutil.move(file_path, error_dest)  # Przeniesienie błędnego pliku
        return False

# Funkcja do podziału danych
def split_data(source_dir, dest_train_dir, dest_val_dir, dest_test_dir, split_ratio=(0.89, 0.01, 0.1)):
    classes = ['Dog', 'Cat']
    min_files = float('inf')  # Minimalna liczba zdjęć w klasie

    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        files = [f for f in os.listdir(class_path) if is_valid_image(os.path.join(class_path, f))]
        min_files = min(min_files, len(files))

    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        files = [f for f in os.listdir(class_path) if is_valid_image(os.path.join(class_path, f))]

        files = files[:min_files]

        num_files = len(files)
        train_size = int(num_files * split_ratio[0])
        val_size = int(num_files * split_ratio[1])

        train_files = files[:train_size]
        val_files = files[train_size:train_size + val_size]
        test_files = files[train_size + val_size:]
        for file_name in train_files:
            os.makedirs(os.path.join(dest_train_dir, class_name), exist_ok=True)
            os.rename(os.path.join(class_path, file_name), os.path.join(dest_train_dir, class_name, file_name))
        for file_name in val_files:
            os.makedirs(os.path.join(dest_val_dir, class_name), exist_ok=True)
            os.rename(os.path.join(class_path, file_name), os.path.join(dest_val_dir, class_name, file_name))
        for file_name in test_files:
            os.makedirs(os.path.join(dest_test_dir, class_name), exist_ok=True)
            os.rename(os.path.join(class_path, file_name), os.path.join(dest_test_dir, class_name, file_name))

# Podział danych
split_data(base_dir, train_dir, validation_dir, test_dir)

# Tworzenie generatorów danych
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  # Normalizacja obrazów
    rotation_range=10,  # Losowy obrót obrazów o maks. 10 stopni
    width_shift_range=0.1,  # Losowe przesunięcie poziome o maks. 10% szerokości
    height_shift_range=0.1,  # Losowe przesunięcie pionowe o maks. 10% wysokości
    shear_range=0.2,  # Losowe przekształcenie perspektywiczne
    zoom_range=0.2,  # Losowe powiększenie/pomniejszenie
    horizontal_flip=True,  # Losowe odbicie w poziomie
    fill_mode='nearest'  # Wypełnianie brakujących pikseli
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)
def create_image_grid(generator, y_true, y_pred, model_type, num_layers, num_neurons):
    """
    Tworzy siatkę obrazów 4x4 (16 obrazów) z losowo wybranymi poprawnymi i błędnymi predykcjami.

    Args:
        generator (ImageDataGenerator): Generator danych.
        y_true (array): Prawdziwe etykiety.
        y_pred (array): Przewidywane etykiety.
        model_type (str): Typ modelu ('FCN' lub 'DRN').
        num_layers (int): Liczba warstw w modelu.
        num_neurons (int): Liczba neuronów w warstwach gęstych.
    """
    # Pobieranie ścieżek i predykcji
    filepaths = generator.filepaths
    correct_indices = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    incorrect_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    # Losowy wybór 8 poprawnych i 8 błędnych przykładów
    random_correct_indices = random.sample(correct_indices, min(8, len(correct_indices)))
    random_incorrect_indices = random.sample(incorrect_indices, min(8, len(incorrect_indices)))

    # Łączenie i losowa kolejność
    selected_indices = random_correct_indices + random_incorrect_indices
    random.shuffle(selected_indices)

    # Wczytanie obrazów i danych
    images = []
    labels = []
    predictions = []

    for idx in selected_indices:
        img_path = filepaths[idx]
        img = Image.open(img_path)
        images.append((img, img_path))
        labels.append(y_true[idx])
        predictions.append(y_pred[idx])

    # Parametry siatki
    grid_size = 4  # Siatka 4x4
    img_size = (150, 150)  # Rozmiar każdego zdjęcia
    padding = 20  # Odstęp między obrazami
    frame_padding = 5  # Odstęp między zdjęciem a ramką
    text_padding = 25  # Odstęp na podpis
    title_padding = 40  # Odstęp na tytuł
    cell_size = (img_size[0] + padding, img_size[1] + padding + text_padding)
    grid_width = grid_size * cell_size[0]
    grid_height = grid_size * cell_size[1] + title_padding

    # Tworzenie obrazu wyjściowego
    output_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(output_image)

    # Ładowanie czcionki
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Dodanie tytułu
    title = f"Network {model_type} prediction example for {num_layers} layers, {num_neurons} neurons"
    draw.text(
        (grid_width // 2, title_padding // 2),
        title,
        fill="black",
        font=title_font,
        anchor="mm"
    )

    # Dodawanie obrazów do siatki
    for i, (img, filepath) in enumerate(images):
        # Przeskalowanie obrazu
        img_resized = img.resize(img_size)
        x = (i % grid_size) * cell_size[0] + padding // 2
        y = (i // grid_size) * cell_size[1] + padding // 2 + title_padding

        # Obliczenie pozycji dla ramki
        frame_x = x - frame_padding
        frame_y = y - frame_padding
        frame_width = img_size[0] + 2 * frame_padding
        frame_height = img_size[1] + 2 * frame_padding

        # Wklejenie obrazu
        output_image.paste(img_resized, (x, y))

        # Ramka wokół zdjęcia
        color = "green" if labels[i] == predictions[i] else "red"
        draw.rectangle(
            [frame_x, frame_y, frame_x + frame_width - 1, frame_y + frame_height - 1],
            outline=color,
            width=3
        )

        # Podpis nazwy pliku
        filename = os.path.basename(filepath)
        draw.text(
            (x, y + img_size[1] + 5),
            filename,
            fill="black",
            font=font,
        )

    # Zapis obrazu
    output_file = os.path.join(
        image_grid_dir, f"grid_{model_type}_{num_layers}_{num_neurons}.png"
    )
    output_image.save(output_file)
    print(f"Saved image grid to {output_file}")
# Funkcja do tworzenia modelu
def create_model(input_shape, num_layers, num_neurons, model_type, learning_rate):
    """
    Tworzy model CNN z określoną liczbą warstw i neuronów.

    Args:
        input_shape (tuple): Kształt wejściowy danych (np. (150, 150, 3)).
        num_layers (int): Liczba warstw gęstych.
        num_neurons (int): Liczba neuronów w pierwszej warstwie gęstej.
        model_type (str): Typ modelu ('FCN' lub 'DRN').
        learning_rate (float): Współczynnik uczenia.

    Returns:
        tf.keras.Model: Skonstruowany i skompilowany model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Warstwy konwolucyjne
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # Warstwy gęste
    model.add(tf.keras.layers.Flatten())
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))
        if model_type == 'DRN':  # Zmniejsz liczba neuronów w przypadku DRN
            num_neurons //= 2

    # Warstwa wyjściowa
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Kompilacja modelu
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'mean_squared_error']
    )

    return model


# Funkcja do trenowania i oceny modelu
def train_and_evaluate(num_layers, num_neurons, learning_rate, model_type):
    """
    Funkcja do trenowania modelu z EarlyStopping, zachowująca wszystkie funkcjonalności
    oryginalnego kodu, takie jak zapis metryk, tworzenie wykresów, macierzy pomyłek i siatek obrazów.

    Args:
        num_layers (int): Liczba warstw gęstych.
        num_neurons (int): Liczba neuronów w warstwach gęstych.
        learning_rate (float): Współczynnik uczenia.
        model_type (str): Typ modelu ('FCN' lub 'DRN').

    Returns:
        tf.keras.Model: Wytrenowany model.
    """
    model = create_model(
        input_shape=(150, 150, 3),
        num_layers=num_layers,
        num_neurons=num_neurons,
        model_type=model_type,
        learning_rate=learning_rate
    )

    # Callback: EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',         # Monitorujemy stratę walidacyjną
        patience=3,                 # Przerwanie po 3 epokach bez poprawy
        restore_best_weights=True,  # Przywrócenie wag z najlepszej epoki
        verbose=1                   # Wyświetlanie informacji o zatrzymaniu
    )

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    metrics = []

    # Trenowanie modelu z EarlyStopping
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,  # Maksymalna liczba epok
        validation_data=validation_generator,
        callbacks=[early_stopping]  # Callback EarlyStopping
    )

    # Zbieranie wyników do zapisania
    for epoch, (train_mse, train_acc, val_mse, val_acc) in enumerate(zip(
        history.history['mean_squared_error'],
        history.history['accuracy'],
        history.history['val_mean_squared_error'],
        history.history['val_accuracy']
    )):
        metrics.append({
            'epoch': epoch + 1,
            'train_mse': train_mse,
            'train_accuracy': train_acc,
            'val_mse': val_mse,
            'val_accuracy': val_acc
        })

    # Testowanie modelu na zbiorze testowym
    test_loss, test_acc, test_mse = model.evaluate(test_generator)
    metrics.append({
        'epoch': 'test',
        'train_mse': None,
        'train_accuracy': None,
        'val_mse': None,
        'val_accuracy': None,
        'test_mse': test_mse,
        'test_accuracy': test_acc
    })

    # Zapisywanie metryk do pliku CSV
    df = pd.DataFrame(metrics)
    csv_file = os.path.join(csv_dir, f'metrics_{num_layers}_layers_{num_neurons}_neurons_{model_type}.csv')
    df.to_csv(csv_file, index=False)

    # Tworzenie wykresów metryk
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(history.history['mean_squared_error']) + 1), history.history['mean_squared_error'], label='Train MSE')
    plt.plot(range(1, len(history.history['val_mean_squared_error']) + 1), history.history['val_mean_squared_error'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title(f'MSE Plot ({model_type}, {num_layers} layers, {num_neurons} neurons)')
    plt.legend()
    plt.savefig(os.path.join(metrics_plot_dir, f'mse_plot_{model_type}_{num_layers}_{num_neurons}.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Train Accuracy')
    plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Plot ({model_type}, {num_layers} layers, {num_neurons} neurons)')
    plt.legend()
    plt.savefig(os.path.join(metrics_plot_dir, f'accuracy_plot_{model_type}_{num_layers}_{num_neurons}.png'))
    plt.close()

    # Generowanie macierzy pomyłek
    Y_pred = model.predict(test_generator)
    y_pred = np.round(Y_pred).astype(int)
    y_true = test_generator.classes

    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=['Cat', 'Dog'])
    print(class_report)

    conf_matrix_file = os.path.join(conf_matrix_dir, f'confusion_matrix_{model_type}_{num_layers}_{num_neurons}.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_type} Matrix ({num_layers} layers, {num_neurons} neurons)')
    plt.savefig(conf_matrix_file)
    plt.close()

    # Tworzenie siatki obrazów
    create_image_grid(test_generator, y_true, y_pred, model_type, num_layers, num_neurons)

    return model

# Parametry sieci
layer_neuron_combinations = [
    (2, 64, 0.001, 'FCN'), (2, 128, 0.001, 'FCN'), (2, 256, 0.001, 'FCN'), (2, 512, 0.001, 'FCN'),
    (2, 64, 0.001, 'DRN'), (2, 128, 0.001, 'DRN'), (2, 256, 0.001, 'DRN'), (2, 512, 0.001, 'DRN'),
    (3, 64, 0.001, 'FCN'), (3, 128, 0.001, 'FCN'), (3, 256, 0.001, 'FCN'), (3, 512, 0.001, 'FCN'),
    (3, 64, 0.001, 'DRN'), (3, 128, 0.001, 'DRN'), (3, 256, 0.001, 'DRN'), (3, 512, 0.001, 'DRN')
]

model_save_dir = os.path.join(base_dir, 'models')
os.makedirs(model_save_dir, exist_ok=True)

# Trenowanie dla różnych konfiguracji
for num_layers, num_neurons, learning_rate, model_type in layer_neuron_combinations:
    print(f"Training model with num_layers={num_layers}, num_neurons={num_neurons}, model_type={model_type}")
    model = train_and_evaluate(num_layers, num_neurons, learning_rate, model_type)
    model_filename = f'model_{num_layers}_layers_{num_neurons}_neurons_{model_type}.h5'
    model.save(os.path.join(model_save_dir, model_filename))
    tf.keras.backend.clear_session()
