import os

def create_dirs(paths):
    """Creates directories if they do not exist."""
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

def validate_image_ext(filename):
    """Check if the file is an image."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    return filename.lower().endswith(valid_extensions)