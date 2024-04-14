from PIL import Image
import torchvision.transforms as v2

# Load your image (make sure to provide the correct path to your image file)
image_path = r'C:\Users\User\Desktop\project_combine\augumentation\backward_another_p1_circle_left010.jpg'  # Replace with your image path
image = Image.open(image_path)

# Calculate the new size, 20% larger than the original
new_width = int(image.width * 1.7)
new_height = int(image.height * 1.7)

# Define the transformation
# resize_transform = transforms.Resize((new_height, new_width))

transforms = v2.Compose([
    # v2.RandomHorizontalFlip(p=0.5),
    v2.RandomResizedCrop(size=(new_height, new_width))
    # v2.RandomCrop(size=(new_height, new_width))
])
# Apply the transformation to the image
resized_image = transforms(image)

# Save or show the resized image
resized_image.save(r'C:\Users\User\Desktop\project_combine\augumentation\resize.jpg')  # Replace with your save path
resized_image.show()
