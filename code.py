import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception, InceptionV3, DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input, Average
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy import stats

# -------------------- DATA PREPARATION -------------------- #
def load_dataset(data_dir, image_size=(299, 299)):
    images, labels = [], []
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")]

    for path in image_paths:
        label_str = path.split('_')[-1].split('.')[0]
        label = int(label_str.rstrip('+'))

        img = load_img(path, target_size=image_size)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(label)

    return np.array(images), np.array(labels)

# -------------------- BASE MODEL CREATION -------------------- #
def create_model(base_model_class, num_classes=4, input_shape=(299, 299, 3)):
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# -------------------- ENSEMBLE MODEL -------------------- #
def create_ensemble(models, num_classes=4, input_shape=(299, 299, 3)):
    inputs = Input(shape=input_shape)
    outputs = [m(inputs) for m in models]
    avg_output = Average()(outputs)
    return Model(inputs, avg_output)

# -------------------- TRAIN & EVALUATE -------------------- #
def evaluate_models(X, y, num_classes=4, k=5, epochs=10):
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    model_classes = {
        "Xception": Xception,
        "GoogLeNet (InceptionV3)": InceptionV3,
        "DenseNet201": DenseNet201
    }

    all_results = {name: [] for name in model_classes}
    all_results["Ensemble"] = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\n==============================")
        print(f"   Fold {fold} / {k}")
        print(f"==============================")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = tf.keras.utils.to_categorical(y[train_idx], num_classes)
        y_test = tf.keras.utils.to_categorical(y[test_idx], num_classes)

        trained_models = {}

        # ---- Train Individual Models ---- #
        for name, base_class in model_classes.items():
            print(f"\n>>> Training {name} <<<")
            model = create_model(base_class, num_classes, input_shape=X.shape[1:])
            model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=4,
                      verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

            _, acc = model.evaluate(X_test, y_test, verbose=0)
            y_pred = np.argmax(model.predict(X_test), axis=1)
            f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average="weighted")

            print(f"Results {name} | Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

            all_results[name].append((acc, f1))
            trained_models[name] = model

        # ---- Ensemble ---- #
        print("\n>>> Evaluating Ensemble (average of all 3) <<<")
        ensemble_model = create_ensemble(list(trained_models.values()), num_classes, input_shape=X.shape[1:])
        ensemble_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        _, acc = ensemble_model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(ensemble_model.predict(X_test), axis=1)
        f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average="weighted")

        print(f"Results Ensemble | Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        all_results["Ensemble"].append((acc, f1))

    # ----------- FINAL RESULTS ----------- #
    print("\n==============================")
    print("   Final Cross-Validation Results")
    print("==============================")
    for name, results in all_results.items():
        accs = [r[0] for r in results]
        f1s = [r[1] for r in results]

        mean_acc, std_acc = np.mean(accs), np.std(accs)
        mean_f1, std_f1 = np.mean(f1s), np.std(f1s)

        # 95% Confidence Interval for Accuracy
        ci_acc = stats.t.interval(0.95, len(accs)-1, loc=mean_acc, scale=stats.sem(accs))

        print(f"\n{name}:")
        print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  95% CI (Accuracy): {ci_acc}")
        print(f"  F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

    # ----------- ANOVA Test (Statistical Significance) ----------- #
    print("\n==============================")
    print("   ANOVA Significance Test (Accuracies)")
    print("==============================")
    densenet_accs = [r[0] for r in all_results["DenseNet201"]]
    xception_accs = [r[0] for r in all_results["Xception"]]
    googlenet_accs = [r[0] for r in all_results["GoogLeNet (InceptionV3)"]]
    ensemble_accs = [r[0] for r in all_results["Ensemble"]]

    f_val, p_val = stats.f_oneway(densenet_accs, xception_accs, googlenet_accs, ensemble_accs)
    print(f"ANOVA F = {f_val:.4f}, p = {p_val:.6f}")

# -------------------- MAIN -------------------- #
if __name__ == "__main__":
    base_dir = "BCI_dataset/HE"
    train_dir = os.path.join(base_dir, "train")

    X, y = load_dataset(train_dir)
    print("Training data shape:", X.shape, y.shape)

    # Run with 5-fold CV and more epochs for actual paper
    evaluate_models(X, y, num_classes=4, k=5, epochs=10)

if __name__ == "__main__":
    base_dir = "BCI_dataset/IHC"
    train_dir = os.path.join(base_dir, "train")

    X, y = load_dataset(train_dir)
    print("Training data shape:", X.shape, y.shape)

    # Run with 5-fold CV and more epochs for actual paper
    evaluate_models(X, y, num_classes=4, k=5, epochs=10)