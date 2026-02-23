import cv2
import os

X = 228
Y = 798
W = 145 
H = 185  

input_root = 'Original_Screenshots'
output_root = 'Sims_Dataset_Clean'

emotions = ['Happy', 'Sad', 'Angry', 'Uncomfortable', 'Tense', 'Embarrassed']

if not os.path.exists(output_root):
    os.makedirs(output_root)

for emotion in emotions:
    input_path = os.path.join(input_root, emotion)
    output_path = os.path.join(output_root, emotion)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if not os.path.exists(input_path):
        continue

    filenames = sorted(os.listdir(input_path))
    
    counter = 1
    for filename in filenames:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(input_path, filename))
            
            if img is not None:
                crop_img = img[Y : Y+H, X : X+W]
                
                if crop_img.size > 0:
                    new_name = f"{emotion.lower()}_{counter:03d}.png"
                    
                    cv2.imwrite(os.path.join(output_path, new_name), crop_img)
                    counter += 1