import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# HoG parametreleri
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# Train klasöründeki paraların HoG özelliklerini ve etiketlerini çıkar
def extract_hog_features(train_folder):
    features = []
    labels = []
    for image_file in os.listdir(train_folder):
        image_path = os.path.join(train_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block, transform_sqrt=True, block_norm="L2-Hys")
        features.append(hog_features)
        labels.append(image_file)  # Dosya adını etiket olarak kullan
    return features, labels

def train_model(train_folder):
    X_train_features, y_train = extract_hog_features(train_folder)
    
    max_length = max(len(feature) for feature in X_train_features)
    X_train = np.array([np.pad(feature, (0, max_length - len(feature)), 'constant') for feature in X_train_features])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_scaled, y_train)
    
    max_features_length = X_train.shape[1]  # X_train'in sütun sayısını al
    
    return model, scaler, max_features_length

def detect_coins(image, model, scaler, max_features_length):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 60)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=300,
                               param1=30, param2=50, minRadius=40, maxRadius=150)
    coins_detected = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        valid_circles = []  # Geçerli dairelerin listesi
        for i in circles[0, :]:
            x, y, r = i
            # Geçersiz daireleri ele
            if x - r < 0 or y - r < 0 or x + r >= gray.shape[1] or y + r >= gray.shape[0] or r < 0:
                continue  # Geçersiz daireyi atla
            valid_circles.append((x, y, r))  # Geçerli daireyi listeye ekle
        # Geçerli daireleri işle
        for x, y, r in valid_circles:
            coin_roi = gray[y-r:y+r, x-r:x+r]
            if coin_roi.shape[0] == 0 or coin_roi.shape[1] == 0:  # Görüntü boşsa veya boyutsuzsa atla
                continue
            resized_roi = cv2.resize(coin_roi, (100, 100))
            hog_features = hog(resized_roi, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
            if len(hog_features) < max_features_length:
                hog_features = np.pad(hog_features, (0, max_features_length - len(hog_features)), 'constant')
            scaled_features = scaler.transform([hog_features])
            predicted_label = model.predict(scaled_features)[0]
            prob = model.predict_proba(scaled_features)[0]
            prob = round(max(prob) * 100, 2)
            cv2.putText(image, f"{predicted_label} ({prob}%)", (x-r, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            coins_detected += 1
    return image, coins_detected



# Klasörleri tanımla
train_folder = "./Train"
test_folder = "./TestV"
output_folder = "./TestV_HoG"

# Modeli eğit
model, scaler, max_features_length = train_model(train_folder)

# Görüntü işleme ve para tespiti için max_features_length değerini kullan
image_files = os.listdir(test_folder)
for image_file in image_files:
    image_path = os.path.join(test_folder, image_file)
    image = cv2.imread(image_path)
    
    result_image, num_coins = detect_coins(image, model, scaler, max_features_length)
    
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, result_image)
    
    print(f"{image_file}: {num_coins} adet para tespit edildi ve {output_folder} klasörüne kaydedildi.")
print("İşlem tamamlandı.")