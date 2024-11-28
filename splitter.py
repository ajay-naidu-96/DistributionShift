import os
import shutil
import random

# Define the path to your dataset
data_dir = "./pokemonclassification/versions/1/PokemonData/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the split ratio (e.g., 80% train, 20% test)
split_ratio = 0.2

# Iterate over each class folder
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)

    if os.path.isdir(class_path):
        # Get the list of images in the class folder
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Shuffle the images
        random.shuffle(images)

        # Calculate the number of test images
        num_test_images = int(len(images) * split_ratio)

        # Create train and test class directories
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Move test images to the test folder
        for image in images[:num_test_images]:
            src = os.path.join(class_path, image)
            dst = os.path.join(test_class_dir, image)
            shutil.move(src, dst)

        # Move the remaining images to the train folder
        for image in images[num_test_images:]:
            src = os.path.join(class_path, image)
            dst = os.path.join(train_class_dir, image)
            shutil.move(src, dst)

print("Dataset split into train and test folders.")
