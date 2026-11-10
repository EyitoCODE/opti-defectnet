"""Random augmentation is confined to the training dataset."""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from splits import CLASSES


class FiberDataset(Dataset):
    def __init__(self, data_dir, is_train=True, samples=None):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.class_to_idx = CLASSES
        if samples is None:
            samples = [p.relative_to(self.data_dir).as_posix()
                       for label in CLASSES for p in sorted((self.data_dir / label).glob("*"))
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        self.image_paths, self.labels = [], []
        for sample in samples:
            path = Path(sample)
            if path.is_absolute() or len(path.parts) != 2 or path.parts[0] not in CLASSES:
                raise ValueError(f"Invalid sample path: {sample}")
            self.image_paths.append(self.data_dir / path)
            self.labels.append(CLASSES[path.parts[0]])
        ops = [transforms.Resize((256, 256))]
        if is_train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(15), transforms.GaussianBlur(kernel_size=3)]
        ops += [transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]
