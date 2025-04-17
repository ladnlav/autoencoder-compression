import torch
import yaml
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import re
import traceback

from model import ConvolutionalAutoencoder
from data_loader import get_mnist_loaders

# Подавляем известные предупреждения
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

def select_device(config_device):
    """Выбирает устройство для вычислений (CPU, CUDA, MPS)."""
    if config_device == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        else: return torch.device("cpu")
    else: return torch.device(config_device)

def load_model(model_type, config, device, specific_latent_dim=None):
    """Загружает модель указанного типа."""
    if model_type == "lossy":
        latent_dim = specific_latent_dim if specific_latent_dim is not None else config["latent_dim_lossy"]
    elif model_type == "lossless":
        latent_dim = config["latent_dim_lossless"]

    num_epochs = config["num_epochs"]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_save_dir = os.path.abspath(os.path.join(project_root, config["model_save_dir"]))
    model_filename = f"{model_type}_autoencoder_ld{latent_dim}_ep{num_epochs}.pth"
    model_path = os.path.join(model_save_dir, model_filename)

    if not os.path.exists(model_path): return None, model_filename

    model = ConvolutionalAutoencoder(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, model_filename

def plot_reconstructions(originals, recons_lossless, recons_lossy, save_path, num_images=10):
    """Отображает и сохраняет сравнение реконструкций."""
    active_rows = []
    if originals is not None: active_rows.append({"title": "Originals", "data": originals.cpu().numpy()})
    if recons_lossless is not None: active_rows.append({"title": "Reconstructed (Lossless)", "data": recons_lossless.cpu().numpy()})
    if recons_lossy is not None: active_rows.append({"title": "Reconstructed (Lossy)", "data": recons_lossy.cpu().numpy()})

    num_rows = len(active_rows)
    if num_rows == 0: return

    plt.figure(figsize=(num_images, num_rows * 1.2))
    for r, row_info in enumerate(active_rows):
        for i in range(num_images):
            ax = plt.subplot(num_rows, num_images, i + 1 + r * num_images)
            img_to_show = np.squeeze(row_info["data"][i])
            plt.imshow(img_to_show, cmap='gray')
            ax.axis('off')
            if i == 0: ax.set_title(row_info["title"], fontsize=10, loc='left', pad=10)

    plt.tight_layout(pad=0.1, h_pad=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Визуализация реконструкций сохранена: {os.path.basename(save_path)}")

def plot_latent_space(vectors, labels, model_name, method, save_path, is_reduced=False):
    """Визуализирует латентные векторы в виде scatter plot."""
    if vectors is None or labels is None: return

    if isinstance(vectors, torch.Tensor): vectors = vectors.cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()

    reduced_vectors = vectors
    if not is_reduced:
        if method == 'tsne': reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300, init='pca', learning_rate='auto')
        elif method == 'pca': reducer = PCA(n_components=2, random_state=42)
        else: return
        try: reduced_vectors = reducer.fit_transform(vectors)
        except Exception as e: print(f"Ошибка снижения размерности {method.upper()} для {model_name}: {e}"); return

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='viridis', s=10, alpha=0.7)
    plt.title(f'Латентное пространство ({model_name}, {method.upper()}) - Scatter Plot')
    plt.xlabel(f'{method.upper()} Component 1'); plt.ylabel(f'{method.upper()} Component 2')
    unique_labels = np.unique(labels)
    legend_labels = [str(int(l)) for l in unique_labels]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(l)), markersize=8) for l in unique_labels]
    plt.legend(handles=handles, labels=legend_labels, title="Цифры")
    plt.grid(True, linestyle='--', alpha=0.5)
    # Создаем директорию перед сохранением, если ее нет
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Scatter plot ({method.upper()}) сохранен: {os.path.relpath(save_path)}") # Печатаем относительный путь для краткости

def plot_latent_manifold(reduced_vectors, reconstructions, model_name, method, save_path, img_zoom=0.3, sample_every=10):
    """Визуализирует многообразие латентного пространства."""
    if reduced_vectors is None or reconstructions is None: return

    reduced_vectors_sampled = reduced_vectors[::sample_every]
    reconstructions_sampled = reconstructions.cpu().numpy()[::sample_every]
    if len(reduced_vectors_sampled) == 0: return

    fig, ax = plt.subplots(figsize=(15, 13))
    x_min, x_max = np.min(reduced_vectors_sampled[:, 0]), np.max(reduced_vectors_sampled[:, 0])
    y_min, y_max = np.min(reduced_vectors_sampled[:, 1]), np.max(reduced_vectors_sampled[:, 1])
    ax.set_xlim(x_min - abs(x_min*0.1), x_max + abs(x_max*0.1)); ax.set_ylim(y_min - abs(y_min*0.1), y_max + abs(y_max*0.1))

    for i in range(len(reduced_vectors_sampled)):
        x, y = reduced_vectors_sampled[i, :]
        img = np.squeeze(reconstructions_sampled[i])
        imagebox = OffsetImage(img, zoom=img_zoom, cmap='gray')
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0)
        ax.add_artist(ab)

    ax.set_title(f'Многообразие латентного пространства ({model_name}, {method.upper()})')
    ax.set_xlabel(f'{method.upper()} Component 1'); ax.set_ylabel(f'{method.upper()} Component 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    # Создаем директорию перед сохранением, если ее нет
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Многообразие ({method.upper()}) сохранено: {os.path.relpath(save_path)}") # Печатаем относительный путь

def find_lossy_models(model_dir, expected_epochs):
    """Ищет файлы lossy моделей с нужным числом эпох."""
    found_models = []
    pattern = re.compile(rf"lossy_autoencoder_ld(\d+)_ep{expected_epochs}\.pth")
    try:
        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match: found_models.append({ "filename": filename, "latent_dim": int(match.group(1)) })
    except Exception as e: print(f"Ошибка поиска моделей в {model_dir}: {e}")
    found_models.sort(key=lambda x: x['latent_dim'])
    return found_models


def visualize(config_path):
    """Основная функция для генерации визуализаций."""
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: print(f"Ошибка чтения конфига {config_path}: {e}"); return

    device = select_device(config.get("device", "auto"))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Базовая директория для всех фигур
    base_results_save_dir = os.path.abspath(os.path.join(project_root, config.get("results_save_dir", "./reports/figures")))
    os.makedirs(base_results_save_dir, exist_ok=True)

    # Загрузка данных
    test_loader = None
    try:
        data_dir = os.path.abspath(os.path.join(project_root, config["data_dir"]))
        _, test_loader = get_mnist_loaders(data_dir=data_dir, batch_size=config["batch_size"], train=False, test=True)
        if test_loader is None: raise ValueError("Test loader is None")
    except Exception as e: print(f"Ошибка загрузки данных: {e}"); return

    # --- 1. Сравнение реконструкций (по конфигу) ---
    model_lossless, _ = load_model("lossless", config, device)
    model_lossy_compare, _ = load_model("lossy", config, device)
    try:
        originals_batch, _ = next(iter(test_loader))
        originals_batch = originals_batch[:15].to(device)
        recons_lossless = model_lossless(originals_batch).detach().cpu() if model_lossless else None
        recons_lossy = model_lossy_compare(originals_batch).detach().cpu() if model_lossy_compare else None
        # Сохраняем в базовую директорию
        save_path = os.path.join(base_results_save_dir, "reconstructions_comparison.png")
        plot_reconstructions(originals_batch, recons_lossless, recons_lossy, save_path, num_images=len(originals_batch))
    except StopIteration: print("Не удалось получить батч для сравнения реконструкций.")
    except Exception as e: print(f"Ошибка при генерации сравнения реконструкций: {e}")

    # --- 2. Поиск и обработка всех найденных lossy моделей ---
    model_save_dir = os.path.abspath(os.path.join(project_root, config["model_save_dir"]))
    found_models_info = find_lossy_models(model_save_dir, config["num_epochs"])

    if not found_models_info: print("Не найдено lossy моделей для детальной визуализации."); return
    print(f"\nНайдено {len(found_models_info)} lossy моделей. Генерация визуализаций латентного пространства...")

    # Загрузка всех найденных моделей
    loaded_models = {}
    for model_info in found_models_info:
        ld = model_info['latent_dim']
        model, _ = load_model("lossy", config, device, specific_latent_dim=ld)
        if model: loaded_models[ld] = model

    if not loaded_models: print("Не удалось загрузить ни одну из найденных lossy моделей."); return

    # Сбор данных (один проход)
    all_data = {ld: {"latents": [], "recons": [], "labels": []} for ld in loaded_models}
    try:
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Сбор данных для визуализации"):
                images, labels_cpu = images.to(device), labels.cpu()
                for ld, model in loaded_models.items():
                    all_data[ld]["latents"].append(model.encode(images).cpu())
                    all_data[ld]["recons"].append(model(images).cpu())
                    all_data[ld]["labels"].append(labels_cpu)
    except Exception as e: print(f"Ошибка при сборе данных: {e}"); return

    # Объединение данных и генерация графиков
    final_data = {}; common_labels = None
    for ld, data_parts in all_data.items():
        try:
            if not data_parts["latents"]: continue
            final_data[ld] = { "latents": torch.cat(data_parts["latents"], dim=0), "recons": torch.cat(data_parts["recons"], dim=0) }
            if common_labels is None: common_labels = torch.cat(data_parts["labels"], dim=0)
        except Exception as e: print(f"Ошибка объединения данных для ld={ld}: {e}")

    if common_labels is None: print("Не удалось собрать метки классов."); return

    # Цикл генерации визуализаций для каждой найденной модели
    for latent_dim, data in final_data.items():
        model_name = f"lossy_ld{latent_dim}"
        print(f"\n--- Генерация графиков для {model_name} ---")
        all_latent_vectors = data["latents"]
        all_reconstructions = data["recons"]

        model_results_dir = os.path.join(base_results_save_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        # Выполнение t-SNE/PCA и построение графиков
        for method in ['tsne', 'pca']:
            reduced_vectors = None
            try:
                if method == 'tsne': reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300, init='pca', learning_rate='auto')
                elif method == 'pca': reducer = PCA(n_components=2, random_state=42)
                reduced_vectors = reducer.fit_transform(all_latent_vectors.numpy())
            except Exception as e: print(f"Ошибка {method.upper()} для {model_name}: {e}")

            if reduced_vectors is not None:
                save_scatter = os.path.join(model_results_dir, f"latent_space_scatter_{method}.png")
                save_manifold = os.path.join(model_results_dir, f"latent_manifold_{method}.png")
                try: plot_latent_space(reduced_vectors, common_labels, model_name, method, save_scatter, is_reduced=True)
                except Exception as e: print(f"Ошибка plot_latent_space ({method}, {model_name}): {e}"); traceback.print_exc()
                try: plot_latent_manifold(reduced_vectors, all_reconstructions, model_name, method, save_manifold, img_zoom=0.35, sample_every=10)
                except Exception as e: print(f"Ошибка plot_latent_manifold ({method}, {model_name}): {e}"); traceback.print_exc()

    print("\nВизуализация завершена.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация визуализаций для автоэнкодеров MNIST.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Путь к файлу конфигурации (YAML).")
    args = parser.parse_args()
    visualize(args.config)