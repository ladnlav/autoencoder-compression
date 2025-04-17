import torch
import torch.nn as nn
import yaml
import os
import argparse
from tqdm import tqdm
import numpy as np

from model import ConvolutionalAutoencoder
from data_loader import get_mnist_loaders
# Убедимся, что select_device доступна
from train import select_device

def evaluate(config_path, project_root=None):
    """
    Основная функция для оценки производительности модели.
    Возвращает словарь с метриками и принимает путь к корню проекта.
    """
    if project_root is None:
        # Предполагаем стандартную структуру: evaluate.py в src/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"(evaluate.py) Корень проекта определен автоматически: {project_root}")
    else:
         print(f"(evaluate.py) Используется переданный корень проекта: {project_root}")


    results = {
        "model_type": None, "latent_dim": None, "model_filename": None,
        "success": False, "error_message": None, "mse_per_image": None,
        "mse_per_pixel": None, "compression_ratio": None
    }

    # 1. Загрузка конфигурации
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Определение типа модели и параметров
    model_type = config["model_type"]
    results["model_type"] = model_type

    if model_type == "lossy": latent_dim = config["latent_dim_lossy"]
    else: latent_dim = config["latent_dim_lossless"]
    results["latent_dim"] = latent_dim

    num_epochs = config["num_epochs"]

    model_save_dir_relative = config["model_save_dir"] # Ожидаем путь типа "./models"
    model_save_dir_abs = os.path.abspath(os.path.join(project_root, model_save_dir_relative))

    # 3. Настройка устройства
    device = select_device(config.get("device", "auto"))

    # 4. Поиск и загрузка модели
    model_filename = f"{model_type}_autoencoder_ld{latent_dim}_ep{num_epochs}.pth"
    results["model_filename"] = model_filename
    model_path = os.path.join(model_save_dir_abs, model_filename)

    model = ConvolutionalAutoencoder(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 5. Загрузка тестовых данных
    data_dir_relative = config["data_dir"] # Ожидаем путь типа "./data"
    data_dir_abs = os.path.abspath(os.path.join(project_root, data_dir_relative))

    _, test_loader = get_mnist_loaders(
        data_dir=data_dir_abs, # Передаем абсолютный путь
        batch_size=config["batch_size"],
        train=False, test=True
    )

    # 6. Вычисление метрик
    total_mse = 0.0; criterion = nn.MSELoss(reduction='sum')
    num_images_processed = 0; num_pixels_per_image = 0
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=f"Оценка {model_type} (ld={latent_dim})", leave=False):
            images = images.to(device)
            if num_pixels_per_image == 0: num_pixels_per_image = images.shape[1]*images.shape[2]*images.shape[3]
            reconstructed_images = model(images)
            batch_mse = criterion(reconstructed_images, images)
            total_mse += batch_mse.item()
            num_images_processed += images.size(0)

    if num_images_processed == 0 or num_pixels_per_image == 0:
            results["error_message"] = "Не удалось обработать изображения для оценки."; return results

    mean_mse_per_image = total_mse / num_images_processed
    mean_mse_per_pixel = total_mse / (num_images_processed * num_pixels_per_image)
    results["mse_per_image"] = mean_mse_per_image
    results["mse_per_pixel"] = mean_mse_per_pixel

    if model_type == "lossy":
        input_dim = num_pixels_per_image
        compression_ratio = input_dim / latent_dim
        results["compression_ratio"] = compression_ratio

    results["success"] = True
    return results

# --- Запуск скрипта как основного ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка сверточного автоэнкодера для MNIST.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Путь к файлу конфигурации (YAML).")
    args = parser.parse_args()
    evaluation_results = evaluate(args.config, project_root=None)

    print("\n--- Результаты оценки (запуск как скрипт) ---")
    if evaluation_results["success"]:
        print(f"Модель: {evaluation_results['model_filename']}")
        print(f"Тип: {evaluation_results['model_type']}")
        print(f"Latent Dim: {evaluation_results['latent_dim']}")
        print(f"Средняя MSE на изображение: {evaluation_results['mse_per_image']:.6f}")
        print(f"Средняя MSE на пиксель:    {evaluation_results['mse_per_pixel']:.8f}")
        if evaluation_results['model_type'] == 'lossy':
            print(f"Коэффициент сжатия: {evaluation_results['compression_ratio']:.2f}x")
        else:
            print("Коэффициент сжатия не рассчитывается для lossless модели.")
    else:
        print("Оценка не удалась.")
        print(f"Ошибка: {evaluation_results['error_message']}")
    print("--------------------------")