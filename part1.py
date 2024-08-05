import cv2
import os
import numpy as np

# Klasörleri tanımla
input_folders = ["Train", "TestV", "TestR"]
output_folders = ["Train_Hough", "TestV_Hough", "TestR_Hough"]

# Klasörleri dolaş
for input_folder, output_folder in zip(input_folders, output_folders):
    # Klasörü kontrol et
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Görüntü dosyalarını al
    image_files = os.listdir(input_folder)
    for image_file in image_files:
        # Görüntüyü yükle
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        # Gri tonlamalı ve bulanıklaştırılmış görüntü oluştur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 2)
        
        # Kenarları tespit et
        edges = cv2.Canny(blurred, 20, 60)
        
        # Hough Daire Dönüşümü kullanarak daireleri bul
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=300,
                                   param1=30, param2=50, minRadius=40, maxRadius=150)

        # Daireleri çiz
        output_image = image.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Dairenin çevresini çiz
                cv2.circle(output_image, (i[0], i[1]), i[2], (0, 255, 0), 4)

        # Sonucu kaydet
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, output_image)

        print(f"{image_file} için daireler tespit edildi ve {output_folder} klasörüne kaydedildi.")

print("İşlem tamamlandı.")