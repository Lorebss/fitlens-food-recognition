from ultralytics import YOLO
from PIL import Image
import os
import shutil

# Dicționar cu mesaje pentru fiecare aliment
info_mesaje = {
    'Pizza': 'Conține multe calorii. Consumă cu moderație.',
    'Pui cu cartofi': 'O masă echilibrată dacă nu e prăjită.',
    'Sarmale': 'Mâncare tradițională consistentă.',
    'Sushi': 'Opțiune ușoară și sănătoasă, bogată în omega-3.',
    'Hamburger': 'Aproximativ echilibrat, dar poate fi gras.',
    'Clatite': 'Desert gustos, dar cu zahăr.',
    'Orez cu legume': 'Opțiune proteică și ușoară.',
    'Gogosi': 'Au mult zahăr și ulei.',
    'Spaghete': 'Bogate în carbohidrați, dar pot fi sănătoase.',
    'Ciorba': 'O alegere ușoară și hidratantă.'
}

# Calea către folderul de predicții
predict_folder = 'runs/detect/predict'

# Șterge folderul predict dacă există
if os.path.exists(predict_folder):
    shutil.rmtree(predict_folder)

# Încarcă modelul YOLO antrenat
model = YOLO('runs/detect/food_model2/weights/best.pt')

# Rulează predicția
results = model.predict(source='sarmale.jpg', save=True, conf=0.5)

# Verifică dacă există detecții
has_detections = Falseq

for r in results:
    if r.boxes is not None and len(r.boxes) > 0:
        has_detections = True

if not has_detections:
    print("⚠️ Nu a fost detectat niciun aliment în imagine.")
else:
    # Afișează mesajele nutriționale
    for r in results:
        for cls_id in r.boxes.cls:
            cls_id = int(cls_id)
            cls_name = model.names[cls_id]
            mesaj = info_mesaje.get(cls_name, "Fără informații despre acest aliment.")
            print(f"🍽️ Detectat: {cls_name} → {mesaj}")

    # Afișează imaginea doar dacă există folderul și fișierul
    if os.path.exists(predict_folder):
        for file in os.listdir(predict_folder):
            if file.endswith('.jpg'):
                result_image_path = os.path.join(predict_folder, file)
                img = Image.open(result_image_path)
                img.show()
                break
