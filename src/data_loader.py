from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_mnist_loaders(data_dir, batch_size, train=True, test=True):
    """
    Создает DataLoader'ы для датасета MNIST из локальных файлов.

    Args:
        data_dir (str): Путь к директории, содержащей папку MNIST/raw с файлами данных.
        batch_size (int): Размер батча.
        train (bool): Создавать ли DataLoader для тренировочного набора.
        test (bool): Создавать ли DataLoader для тестового набора.

    Returns:
        tuple: Кортеж с (train_loader, test_loader). Если какой-то из них не запрошен,
               на его месте будет None.
    """
    # Стандартная трансформация для MNIST:
    # 1. Преобразование изображения в тензор PyTorch.
    # 2. Нормализация значений пикселей к диапазону [0, 1].
    #    Для автоэнкодеров часто удобнее [0, 1], особенно если последняя активация - Sigmoid.
    transform = transforms.Compose([
        transforms.ToTensor(), # Преобразует PIL Image/ndarray (H x W x C) в тензор (C x H x W) и масштабирует в [0.0, 1.0]
    ])

    train_loader = None
    test_loader = None

    mnist_root = os.path.join(data_dir, 'MNIST')

    if train:
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=False,
            transform=transform
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    if test:
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=False,
            transform=transform
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    return train_loader, test_loader

if __name__ == '__main__':
    DATA_DIR = r"data"
    BATCH_SIZE = 64

    train_loader, test_loader = get_mnist_loaders(DATA_DIR, BATCH_SIZE)

    if train_loader:
        print(f"Тренировочный загрузчик создан. Количество батчей: {len(train_loader)}")
        # Получим один батч для проверки
        images, labels = next(iter(train_loader))
        print(f"Размер батча изображений: {images.shape}") # Ожидаем [batch_size, 1, 28, 28]
        print(f"Размер батча меток: {labels.shape}")
        print(f"Тип данных изображений: {images.dtype}") # Ожидаем torch.float32
        print(f"Диапазон значений пикселей: min={images.min()}, max={images.max()}") # Ожидаем [0.0, 1.0]


    if test_loader:
        print(f"\nТестовый загрузчик создан. Количество батчей: {len(test_loader)}")
        images, labels = next(iter(test_loader))
        print(f"Размер батча изображений: {images.shape}")