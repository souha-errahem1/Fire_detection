import cv2
from ultralytics import YOLO
import os
import numpy as np
from PIL import Image

# Charger le modèle
model = YOLO('model.pt')

# Spécifier le chemin du dossier contenant les images
folder_path = r'D:\fire detection\pythonProject\images'  # Remplacez par votre chemin de dossier

# Vérifier si le dossier existe
if not os.path.exists(folder_path):
    print("Dossier non trouvé à l'emplacement spécifié!")
    exit()

# Parcourir toutes les images dans le dossier
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Ajouter d'autres formats si nécessaire
        image_path = os.path.join(folder_path, filename)

        # Charger l'image
        try:
            image_pil = Image.open(image_path)  # Ouvrir l'image avec PIL
            print(f"Traitement de l'image: {filename} avec le mode: {image_pil.mode}")
            image = np.array(image_pil)  # Convertir l'image en tableau NumPy

            # S'assurer que l'image est un tableau NumPy contigu
            image = np.ascontiguousarray(image)

            # Convertir l'image en BGR si nécessaire
            if image_pil.mode == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image_pil.mode == 'RGBA':
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image_pil.mode == 'L':  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                print(f"Format d'image non supporté pour: {filename}")
                continue  # Passer à la prochaine image

            # Redimensionner si nécessaire
            image_resized = cv2.resize(image, (640, 480))

            # Faire des prédictions
            results = model(image_resized)

            # Traiter les résultats
            for info in results:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf.item()  # Obtenir la confiance comme un scalaire
                    if confidence > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convertir en entiers
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

            # Afficher le résultat
            cv2.imshow('Detected Image', image)
            cv2.waitKey(0)  # Attendre une touche pour passer à l'image suivante

        except Exception as e:
            print(f"Erreur lors du traitement de l'image {filename}: {e}")

cv2.destroyAllWindows()
