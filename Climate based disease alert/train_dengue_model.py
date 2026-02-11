import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

#load dataset
df = pd.read_csv("D:\\Climate based disease alert\\data\\climate_disease_dataset.csv")
#feature selection
features = [
    "avg_temp_c",
    "precipitation_mm",
    "air_quality_index",
    "uv_index",
    "population_density"
]
TARGET = "dengue_cases"
x = df[features]
y_raw = df[TARGET]
#converting to risk levels
def dengue_risk_level(cases):
    if cases < 50:
        return "LOW"
    elif cases < 200:
        return "MEDIUM"
    else:
        return "HIGH"
y = y_raw.apply(dengue_risk_level)
# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
#train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
#assigning weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)
#feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
#building neural networks
model = Sequential([
    Dense(32, activation="relu", input_shape=(x_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
#train model
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
model.fit(
    x_train_scaled,
    y_train,
    validation_data=(x_test_scaled, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)
#saving the model
model.save("dengue_nn_model.keras")
joblib.dump(scaler, "dengue_scaler.pkl")
joblib.dump(label_encoder, "dengue_label_encoder.pkl")
print("Dengue model training complete")
print("Saved:")
print(" - dengue_nn_model.keras")
print(" - dengue_scaler.pkl")
print(" - dengue_label_encoder.pkl")
