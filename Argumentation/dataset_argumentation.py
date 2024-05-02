import cv2
import os
import albumentations as A

# Define the transformation
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.75),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.75)
])

# Directories setup
folder = 'train'
image_dir = r"C:\Users\eofeh\Desktop\Model\datasets\%s\images" %(folder)
label_dir = r"C:\Users\eofeh\Desktop\Model\datasets\%s\labels" %(folder)
augmented_image_dir = r"C:\Users\eofeh\Desktop\Model\datasets\%s\%s_augmented_images" %(folder, folder)
augmented_label_dir = r"C:\Users\eofeh\Desktop\Model\datasets\%s\%s_augmented_labels" %(folder, folder)

os.makedirs(augmented_image_dir, exist_ok=True)
os.makedirs(augmented_label_dir, exist_ok=True)

# Process each image in the image directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        for i in range(7):
        # Apply the transformation
            transformed_image = transform(image=image)['image']
            
            # Save the augmented image
            augmented_image_path = os.path.join(augmented_image_dir, f'{filename[:-4]}_aug_{i+1}.jpg')
            cv2.imwrite(augmented_image_path, transformed_image)
            
            # Copy label file to augmented labels directory with corresponding naming
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = f.read()
                augmented_label_path = os.path.join(augmented_label_dir, f'{filename[:-4]}_aug_{i+1}.txt')
                with open(augmented_label_path, 'w') as f:
                    f.write(labels)

            print(f'Processed and saved augmented image and label for {filename}')
