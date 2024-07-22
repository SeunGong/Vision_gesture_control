import cv2
import albumentations as A

# Assuming you are reading an image correctly
image_path = 'Argumentation/image.jpg'
# image_path = r"C:\Users\eofeh\Desktop\Model\datasets_copy\test\images"
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Unable to read image at path {image_path}")

# Define a transformation or a composition of transformations
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.75),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.75)
])

for i in range(1,9):
    # Apply transformation
    transformed_image = transform(image=image)['image']
    output_path = f'Argumentation/image_augmented_{i}.jpg'
    cv2.imwrite(output_path, transformed_image)
    print(f'Saved: {output_path}')

