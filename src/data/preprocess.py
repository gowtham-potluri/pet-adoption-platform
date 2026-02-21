import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
from PIL import Image


def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False


def split_data(input_dir: str, output_dir: str, params_path="params.yaml"):

    # Load parameters
    with open(params_path) as f:
        params = yaml.safe_load(f)

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    cats_path = os.path.join(input_dir, "Cat")
    dogs_path = os.path.join(input_dir, "Dog")

    cat_images = []
    dog_images = []

    for f in os.listdir(cats_path):
        path = os.path.join(cats_path, f)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and is_valid_image(path):
            cat_images.append(path)

    for f in os.listdir(dogs_path):
        path = os.path.join(dogs_path, f)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and is_valid_image(path):
            dog_images.append(path)

    print("Valid Cats:", len(cat_images))
    print("Valid Dogs:", len(dog_images))

    images = cat_images + dog_images
    labels = ["cat"] * len(cat_images) + ["dog"] * len(dog_images)

    # 80% train, 20% temp (controlled by params.yaml)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    # Split temp equally into val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state,
    )

    # Create folders and copy files
    for split, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        for img_path, label in zip(X, y):
            dest = os.path.join(output_dir, split, label)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(img_path, dest)

    print("Data split complete!")