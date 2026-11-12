import os
import random
import shutil

# Script care organizează datasetul în două subfoldere: train și val, necesare pentru antrenarea modelului YOLO.

SOURCE_IMAGES = 'food-dataset/images'
SOURCE_LABELS = 'food-dataset/labels'
DEST_FOLDER = 'food-dataset'

TRAIN_RATIO = 0.8

# Creează structura corectă
os.makedirs(os.path.join(DEST_FOLDER, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(DEST_FOLDER, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(DEST_FOLDER, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(DEST_FOLDER, 'val', 'labels'), exist_ok=True)

# Obține fișierele de imagine
image_files = [f for f in os.listdir(SOURCE_IMAGES) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

train_count = int(len(image_files) * TRAIN_RATIO)
train_images = image_files[:train_count]
val_images = image_files[train_count:]

# Mută imaginile și etichetele
def move_files(files, split):
    for img_file in files:
        label_file = os.path.splitext(img_file)[0] + '.txt'

        shutil.copy2(os.path.join(SOURCE_IMAGES, img_file), os.path.join(DEST_FOLDER, split, 'images', img_file))
        if os.path.exists(os.path.join(SOURCE_LABELS, label_file)):
            shutil.copy2(os.path.join(SOURCE_LABELS, label_file), os.path.join(DEST_FOLDER, split, 'labels', label_file))

move_files(train_images, 'train')
move_files(val_images, 'val')

print('✅ Dataset împărțit în train și val cu succes!')
