import torch
import torch.nn as nn

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        """
        Инициализация сверточного автоэнкодера.

        Args:
            latent_dim (int): Размерность латентного пространства.
        """
        super(ConvolutionalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # --- Энкодер ---
        # Вход: [batch_size, 1, 28, 28]
        self.encoder = nn.Sequential(
            # Свертка 1: 1x28x28 -> 16x14x14
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            # padding=1 при kernel_size=3, stride=2 сохраняет размер при нечетном входе,
            # но уменьшает вдвое при четном: floor((28 + 2*1 - 3)/2 + 1) = floor(27/2 + 1) = 13 + 1 = 14
            nn.ReLU(),
            # Свертка 2: 16x14x14 -> 32x7x7
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            # floor((14 + 2*1 - 3)/2 + 1) = floor(13/2 + 1) = 6 + 1 = 7
            nn.ReLU(),
            # Выпрямление: 32x7x7 -> 32 * 7 * 7 = 1568
            nn.Flatten(),
            # Полносвязный слой для получения латентного вектора
            nn.Linear(32 * 7 * 7, latent_dim)
        )

        # --- Декодер ---
        self.decoder = nn.Sequential(
            # Полносвязный слой для восстановления размера перед свертками
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            # Восстановление формы тензора: 1568 -> [batch_size, 32, 7, 7]
            nn.Unflatten(dim=1, unflattened_size=(32, 7, 7)),
            # Транспонированная свертка 1: 32x7x7 -> 16x14x14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # H_out = (H_in - 1)*stride - 2*padding + kernel_size + output_padding
            # H_out = (7 - 1)*2 - 2*1 + 3 + 1 = 6*2 - 2 + 3 + 1 = 12 - 2 + 3 + 1 = 14
            nn.ReLU(),
            # Транспонированная свертка 2: 16x14x14 -> 1x28x28
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # H_out = (14 - 1)*2 - 2*1 + 3 + 1 = 13*2 - 2 + 3 + 1 = 26 - 2 + 3 + 1 = 28
            nn.Sigmoid() # Сигмоида, чтобы выходные значения были в диапазоне [0, 1], как и входные нормализованные данные
        )

    def forward(self, x):
        """
        Прямой проход данных через автоэнкодер.

        Args:
            x (torch.Tensor): Входной тензор изображений [batch_size, 1, 28, 28].

        Returns:
            torch.Tensor: Реконструированный тензор изображений [batch_size, 1, 28, 28].
        """
        latent_representation = self.encoder(x)
        reconstructed_x = self.decoder(latent_representation)
        return reconstructed_x

    def encode(self, x):
        """
        Кодирует входные данные в латентное представление.
        Args:
            x (torch.Tensor): Входной тензор изображений.
        Returns:
            torch.Tensor: Латентное представление.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Декодирует латентное представление обратно в данные.
        Args:
            z (torch.Tensor): Латентное представление.
        Returns:
            torch.Tensor: Реконструированный тензор изображений.
        """
        return self.decoder(z)

if __name__ == '__main__':
    latent_dim_test = 64
    model = ConvolutionalAutoencoder(latent_dim=latent_dim_test)
    print("Структура модели:")
    print(model)

    batch_size_test = 4
    dummy_input = torch.randn(batch_size_test, 1, 28, 28)

    print(f"\nРазмер входных данных: {dummy_input.shape}")

    # Прогоним через энкодер
    latent_code = model.encode(dummy_input)
    print(f"Размер латентного кода: {latent_code.shape}") # Ожидаем [batch_size_test, latent_dim_test]

    # Прогоним через декодер
    reconstruction = model.decode(latent_code)
    print(f"Размер реконструированных данных: {reconstruction.shape}") # Ожидаем [batch_size_test, 1, 28, 28]

    # Проверим полный проход
    output = model(dummy_input)
    print(f"Размер выходных данных (полный проход): {output.shape}") # Ожидаем [batch_size_test, 1, 28, 28]
    print(f"Диапазон значений на выходе: min={output.min().item()}, max={output.max().item()}") # Ожидаем [0, 1] из-за Sigmoid