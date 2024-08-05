import cv2
import os
import numpy as np
from sklearn import svm
from skimage.feature import hog
import joblib

# HoG parametreleri
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

def extract_hog_features(image):
    # Gri tonlamalı görüntüyü oluştur ve sabit boyuta getir
    image = cv2.resize(image, (128, 128))  # Örnek olarak 128x128 kullanıldı
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # HoG özelliklerini çıkar
    features = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
    return features

def prepare_training_data(data_folder):
    X = []
    y = []

    for image_file in os.listdir(data_folder):
        image_path = os.path.join(data_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            continue  # Bozuk ya da okunamayan dosyaları atla

        # HoG özelliklerini çıkar
        features = extract_hog_features(image)
        
        # Özellik vektörünü listeye ekle
        X.append(features)
        
        # Etiketi listeye ekle
        label = image_file.split("_")[0]
        y.append(label)
    
    # X ve y'yi numpy array'lerine dönüştür
    X = np.array(X)
    y = np.array(y)

    return X, y

def train_svm(X_train, y_train):
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

def count_and_identify_coins(image_path, model):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improved Gaussian blur for noise reduction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Enhanced adaptive thresholding for better edge detection
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Fine-tuned morphological opening to clean up small artifacts
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Dilate to connect adjacent objects
    dilated = cv2.dilate(opening, kernel, iterations=1)

    # Find contours with refined criteria for coin detection
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coin_dict = {}
    for contour in contours:
        area = cv2.contourArea(contour)
        # Adjust these thresholds based on your empirical observation of coin sizes
        if 100 < area < 10000:  # Example thresholds, adjust accordingly
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            coin = cv2.bitwise_and(image, image, mask=mask)

            hog_features = extract_hog_features(coin)
            prediction = model.predict(hog_features.reshape(1, -1))
            predicted_class = prediction[0]

            if predicted_class in coin_dict:
                coin_dict[predicted_class] += 1
            else:
                coin_dict[predicted_class] = 1

    return coin_dict

def draw_offsets_and_segmentation(image_path, output_path, coin_dict):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Para bilgilerini yazdır
    idx_offset = 0
    for idx, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)

            # Her para için metni çizin
            text = f"{list(coin_dict.keys())[idx]}: {list(coin_dict.values())[idx]}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = int(cX - text_size[0] / 2)
            text_y = int(cY + text_size[1] / 2)
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            idx_offset += 30  # Bir sonraki yazı için offset ayarla

    # Segmentasyon haritasını çiz
    segmentation_map = np.zeros_like(image)  # Renkli bir görüntü oluştur
    cv2.drawContours(segmentation_map, contours, -1, (120, 255, 120), 1)  # Beyaz renkte çiz

    # Resmin sağ üst köşesine segmentasyon haritasını büyükçe çizdir
    height, width = image.shape[:2]
    segmentation_map_resized = cv2.resize(segmentation_map, (width // 5, height // 5))

    offset_x = width - segmentation_map_resized.shape[1] - 20
    offset_y = 20
    image[offset_y:offset_y+segmentation_map_resized.shape[0], offset_x:offset_x+segmentation_map_resized.shape[1]] = segmentation_map_resized

    cv2.imwrite(output_path, image)

def main():
    # Eğitim verilerini hazırla
    X_train, y_train = prepare_training_data("./Train")
    print("Eğitim verisi hazır.")

    # SVM modelini eğit
    svm_classifier = train_svm(X_train, y_train)
    print("SVM modeli eğitildi.")

    # Eğitilmiş modeli kaydet
    joblib.dump(svm_classifier, "./TrainingDataFolder/coin_detector_model.pkl")
    print("Eğitilmiş model kaydedildi.")

    # "TestV" klasöründeki her bir görüntü için işlem yap
    test_images_folder = "./TestV"
    output_folder = "./TestV_HoG"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(test_images_folder):
        image_path = os.path.join(test_images_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # Madeni paraları sayıp tanımla
        coin_dict = count_and_identify_coins(image_path, svm_classifier)

        # Madeni paraların ofsetlerini ve segmentasyon haritasını çiz
        draw_offsets_and_segmentation(image_path, output_path, coin_dict)

        # Sonuçları yazdır
        print(f"{image_file}: {coin_dict}")

if __name__ == "__main__":
    main()