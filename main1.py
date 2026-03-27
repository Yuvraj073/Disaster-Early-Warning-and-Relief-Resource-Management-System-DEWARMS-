#!/usr/bin/env python3

import os
import argparse
import glob
import sys
import itertools
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


def load_dataset(data_dir, img_size=100, exts=("*.jpg", "*.png")):
    images = []
    labels = []
    data_dir = os.path.abspath(data_dir)

    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for ext in exts:
            pattern = os.path.join(folder_path, ext)
            for img_path in glob.glob(pattern):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img.astype(np.float32))
                labels.append(folder)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    return images, labels


def build_custom_cnn(input_shape=(100, 100, 3), num_classes=5):
    model = Sequential([
        InputLayer(input_shape=input_shape),

        Conv2D(25, (5, 5), activation='relu', padding='same'),
        MaxPool2D((2, 2), padding='same'),

        Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'),
        MaxPool2D((2, 2), padding='same'),
        BatchNormalization(),

        Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'),
        MaxPool2D((2, 2), padding='valid'),
        BatchNormalization(),

        Flatten(),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    return model


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        disp = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        disp = cm

    print(disp)
    plt.figure(figsize=(8, 6))
    plt.imshow(disp, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = disp.max() / 2.
    for i, j in itertools.product(range(disp.shape[0]), range(disp.shape[1])):
        plt.text(j, i, format(disp[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if disp[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def train(args):
    print("Loading dataset from:", args.data_dir)
    X, y = load_dataset(args.data_dir, img_size=args.img_size)
    print("Loaded", X.shape[0], "images.")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Save class names so we can use them during prediction
    class_file = args.save_model + "_classes.npy"
    np.save(class_file, class_names)
    print("Saved class names to:", class_file)

    # ----- Train / Val / Test split -----
    total_val_test = args.val_split + args.test_split
    if total_val_test >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    # First: split into train and temp (val+test) with stratify
    x_train, x_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        test_size=total_val_test,
        random_state=42,
        stratify=y_encoded
    )

    # Second: split temp into val and test
    test_fraction_within_temp = args.test_split / total_val_test

    # Check if every class in y_temp has at least 2 samples
    unique_temp, counts_temp = np.unique(y_temp, return_counts=True)
    print("Temp (val+test) class counts:", dict(zip(unique_temp, counts_temp)))

    if np.min(counts_temp) < 2:
        print("Warning: some classes in temp set have only 1 sample. "
              "Disabling stratify for val/test split.")
        stratify_second = None
    else:
        stratify_second = y_temp

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=test_fraction_within_temp,
        random_state=42,
        stratify=stratify_second
    )

    print(f"Train size: {x_train.shape[0]}")
    print(f"Val size:   {x_val.shape[0]}")
    print(f"Test size:  {x_test.shape[0]}")

    # Normalize
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0

    # One-hot labels for train/val
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    input_shape = (args.img_size, args.img_size, 3)
    model = build_custom_cnn(input_shape=input_shape, num_classes=num_classes)

    opt = Adam(learning_rate=args.lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=args.rotation,
        width_shift_range=args.width_shift,
        height_shift_range=args.height_shift,
        zoom_range=args.zoom,
        horizontal_flip=args.hflip,
        vertical_flip=args.vflip,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    # Callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience=5,
        factor=0.5,
        verbose=1,
        min_lr=1e-6
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        args.save_model,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    steps_per_epoch = max(1, x_train.shape[0] // args.batch_size)

    # Training
    history = model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(x_val, y_val_cat),
        steps_per_epoch=steps_per_epoch,
        callbacks=[reduce_lr, early_stop, checkpoint],
        verbose=1
    )

    # Save final model (best weights already saved by checkpoint)
    model.save(args.save_model)
    print("Model saved to:", args.save_model)

    # ---- Evaluation on validation set ----
    print("\nEvaluating on validation set...")
    y_val_pred_probs = model.predict(x_val)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    cm_val = confusion_matrix(y_val, y_val_pred)
    plot_confusion_matrix(cm_val, classes=class_names, title='Confusion matrix (Validation)')

    # ---- Evaluation on test set ----
    print("\nEvaluating on test set...")
    y_test_pred_probs = model.predict(x_test)
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)
    cm_test = confusion_matrix(y_test, y_test_pred)
    plot_confusion_matrix(cm_test, classes=class_names, title='Confusion matrix (Test)')

    test_accuracy = np.mean(y_test_pred == y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    frac_error = 1.0 - np.diag(cm_test) / np.sum(cm_test, axis=1)
    plt.figure()
    plt.bar(np.arange(len(frac_error)), frac_error)
    plt.xticks(np.arange(len(frac_error)), class_names, rotation=45)
    plt.xlabel("True label")
    plt.ylabel("Fraction misclassified (Test set)")
    plt.tight_layout()
    plt.show()


def predict(args):
    # Load trained model
    model = load_model(args.model)

    # Load saved class names
    class_file = args.model + "_classes.npy"
    if not os.path.exists(class_file):
        raise FileNotFoundError(f"Class file not found: {class_file}")
    classes = np.load(class_file, allow_pickle=True)

    # Read and preprocess image
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Cannot read image")

    img = cv2.resize(img, (args.img_size, args.img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))

    print("Predicted class index:", idx)
    print("Predicted class name:", classes[idx])
    print("Probabilities:", probs)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)

    # Train sub-command
    p_train = sub.add_parser('train')
    p_train.add_argument('--data_dir', required=True)
    p_train.add_argument('--save_model', default='in_estimator_2.model')
    p_train.add_argument('--img_size', type=int, default=100)
    p_train.add_argument('--epochs', type=int, default=100)
    p_train.add_argument('--batch_size', type=int, default=8)
    # Now we have both val and test splits
    p_train.add_argument('--val_split', type=float, default=0.1)
    p_train.add_argument('--test_split', type=float, default=0.1)
    p_train.add_argument('--lr', type=float, default=0.001)
    p_train.add_argument('--rotation', type=float, default=180.0)
    p_train.add_argument('--width_shift', type=float, default=0.1)
    p_train.add_argument('--height_shift', type=float, default=0.1)
    p_train.add_argument('--zoom', type=float, default=0.1)
    p_train.add_argument('--hflip', action='store_true', default=True)
    p_train.add_argument('--vflip', action='store_true', default=True)

    # Predict sub-command
    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--model', required=True)
    p_pred.add_argument('--image', required=True)
    p_pred.add_argument('--img_size', type=int, default=100)

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)


if __name__ == '__main__':
    main()
