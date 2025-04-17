import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import argparse
from tqdm import tqdm

from model import ConvolutionalAutoencoder
from data_loader import get_mnist_loaders

def select_device(config_device):
    """Выбирает устройство для вычислений (CPU, CUDA, MPS)."""
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(config_device)

def train(config_path):
    """Основная функция для запуска обучения модели."""
    # 1. Загрузка конфигурации
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_type = config.get("model_type")
    print(f"Запуск обучения для {model_type}.")

    # 2. Настройка устройства
    device = select_device(config.get("device", "auto"))
    print(f"Используемое устройство: {device}")

    # 3. Загрузка данных
    train_loader, _ = get_mnist_loaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        train=True,
        test=False
    )
    print(f"Данные загружены. Размер батча: {config['batch_size']}.")

    # 4. Инициализация модели
    if model_type == "lossy":
        latent_dim = config["latent_dim_lossy"]
    else: # lossless
        latent_dim = config["latent_dim_lossless"]

    model = ConvolutionalAutoencoder(latent_dim=latent_dim).to(device)
    print(f"Модель инициализирована с latent_dim = {latent_dim}.")

    # 5. Функция потерь и оптимизатор
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    print(f"Оптимизатор: Adam, Learning Rate: {config['learning_rate']}")

    # 6. Цикл обучения
    num_epochs = config["num_epochs"]
    print(f"Начало обучения на {num_epochs} эпох...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs}", leave=False)

        for images, _ in progress_bar:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Эпоха {epoch+1}/{num_epochs}, train MSE: {epoch_loss:.6f}")

    print("Обучение завершено.")

    # 7. Сохранение модели
    save_dir = config["model_save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    model_filename = f"{model_type}_autoencoder_ld{latent_dim}_ep{num_epochs}.pth"
    save_path = os.path.join(save_dir, model_filename)

    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена в {save_path}")


# --- Запуск скрипта ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение сверточного автоэнкодера для MNIST.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к файлу конфигурации (YAML)."
    )
    args = parser.parse_args()

    train(args.config)