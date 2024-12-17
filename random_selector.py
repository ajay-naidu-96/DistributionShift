import os
import shutil
import random

def copy_random_files(src_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Walk through all subdirectories in the source directory
    for subdir, _, files in os.walk(src_dir):
        # Skip empty directories
        if not files:
            continue
        
        # Pick a random file from the list of files in the subdirectory
        random_file = random.choice(files)
        
        # Get the full path of the source file
        src_file = os.path.join(subdir, random_file)
        
        # Get the destination path (we want to keep the same file name in the destination)
        dest_file = os.path.join(dest_dir, random_file)
        
        # Copy the selected file to the destination directory
        shutil.copy(src_file, dest_file)
        print(f"Copied: {src_file} -> {dest_file}")

# Example usage
source_directory = './Data/cmnist/fixed/7/'  # Change this to your source directory path
destination_directory = './test/'  # Change this to your destination directory path

copy_random_files(source_directory, destination_directory)
