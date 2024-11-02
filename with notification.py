from ultralytics import YOLO
import cvzone
import cv2
import math
from twilio.rest import Client

# Informations d'authentification Twilio

# Fonction pour envoyer la notification SMS
from twilio.rest import Client

def send_sms():
    account_sid = 'US06d6c21039e02092a217674b578a9cd4'  # Remplacez par votre Account SID
    auth_token = 'f9efc2b846fa57a6e2dade3afe01fa4f'      # Remplacez par votre Auth Token
    client = Client(account_sid, auth_token)

    try:
        message = client.messages.create(
            body="Alerte : Une flamme a été détectée !",
            from_='+12702006468',  # Remplacez par votre numéro Twilio
            to='+21629841009'       # Remplacez par le numéro de destination
        )
        print(f"Message sent: {message.sid}")
    except Exception as e:
        print(f"Erreur d'envoi de SMS: {str(e)}")

# Appelez cette fonction après la détection de la flamme


# Running real time from webcam
cap = cv2.VideoCapture('D:/fire detection/pythonProject/fire4.mp4')
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo")

model = YOLO('D:/fire detection/pythonProject/fire.pt')

# Reading the classes
classnames = ['fire']

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Arrête la boucle si la vidéo est terminée
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Getting bbox, confidence, and class names informations to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])

            if confidence > 50:  # Si la confiance est supérieure à 50%
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

                # Notification SMS si une flamme est détectée
                if classnames[Class] == 'fire':
                    send_sms()  # Appelle la fonction pour envoyer le SMS

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Quitte la boucle si 'q' est pressé

cap.release()
cv2.destroyAllWindows()
