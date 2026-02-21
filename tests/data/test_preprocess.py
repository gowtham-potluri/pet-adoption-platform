import os
from PIL import Image
from src.data.preprocess import split_data


def create_dummy_image(path):
    img = Image.new("RGB", (10, 10), color="red")
    img.save(path)

def test_split_data(tmp_path):

    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    cat_dir = raw_dir / "Cat"
    dog_dir = raw_dir / "Dog"
    cat_dir.mkdir(parents=True)
    dog_dir.mkdir(parents=True)

    for i in range(20):
        create_dummy_image(cat_dir / f"cat{i}.jpg")
        create_dummy_image(dog_dir / f"dog{i}.jpg")

    split_data(str(raw_dir), str(output_dir))

    assert os.path.exists(output_dir / "train")
    assert os.path.exists(output_dir / "val")
    assert os.path.exists(output_dir / "test")
