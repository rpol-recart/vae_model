import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split # для разделения на трейн/валидацию
import numpy as np
import matplotlib.pyplot as plt # для визуализации прогресса

# --- Вставляем код VAE и Encoder/Decoder отсюда (из предыдущего ответа) ---
class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim, sequence_length):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        with torch.no_grad():
            dummy_input_shape = (1, input_channels, sequence_length)
            dummy_input = torch.randn(*dummy_input_shape)
            conv_out_shape = self._forward_conv(dummy_input).shape
            self.flattened_size = conv_out_shape[1] * conv_out_shape[2]
        self.fc_hidden = nn.Linear(self.flattened_size, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

    def _forward_conv(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc_hidden(x))
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels, sequence_length, encoder_flattened_size, encoder_conv_out_shape_spatial):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_flattened_size = encoder_flattened_size
        self.encoder_conv_out_channels = 128
        self.encoder_conv_out_spatial_dim = encoder_conv_out_shape_spatial
        self.fc_decode_hidden = nn.Linear(latent_dim, 256)
        self.fc_decode = nn.Linear(256, self.encoder_flattened_size)
        self.t_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_t1 = nn.BatchNorm1d(64)
        self.t_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_t2 = nn.BatchNorm1d(32)
        self.t_conv3 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn_t3 = nn.BatchNorm1d(16)
        self.t_conv4 = nn.ConvTranspose1d(16, output_channels, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, z):
        x = F.leaky_relu(self.fc_decode_hidden(z))
        x = F.leaky_relu(self.fc_decode(x))
        x = x.view(x.size(0), self.encoder_conv_out_channels, self.encoder_conv_out_spatial_dim)
        x = F.leaky_relu(self.bn_t1(self.t_conv1(x)))
        x = F.leaky_relu(self.bn_t2(self.t_conv2(x)))
        x = F.leaky_relu(self.bn_t3(self.t_conv3(x)))
        reconstructed_x = self.t_conv4(x)
        return reconstructed_x

class VAE(nn.Module):
    def __init__(self, input_channels=2, latent_dim=32, sequence_length=512):
        super(VAE, self).__init__()
        self.input_channels = input_channels
        self.encoder = Encoder(input_channels, latent_dim, sequence_length)
        encoder_flattened_size = self.encoder.flattened_size
        encoder_conv_out_spatial_dim = sequence_length // (2**4)
        self.decoder = Decoder(latent_dim, input_channels, sequence_length, encoder_flattened_size, encoder_conv_out_shape_spatial)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

    def get_latent_features(self, x):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu

def loss_function_vae(recon_x, x, mu, log_var, beta=1.0):
    min_len = min(recon_x.size(2), x.size(2))
    recon_x_trimmed = recon_x[:, :, :min_len]
    x_trimmed = x[:, :, :min_len]
    recon_loss = F.mse_loss(recon_x_trimmed, x_trimmed, reduction='sum') / x.size(0)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return recon_loss + beta * kld_loss
# --- Конец вставки кода VAE ---


# 1. Dataset
class DTAHeatDataset(Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list (list): Список тензоров или numpy массивов.
                              Каждый элемент должен иметь форму (num_channels, sequence_length).
                              Для нашего VAE num_channels = 2.
        """
        self.data_list = data_list
        if not data_list:
            raise ValueError("data_list не может быть пустым.")
        
        # Проверка размерности первого элемента для consistency
        # (предполагаем, что все элементы имеют одинаковую размерность каналов и длины)
        first_item_shape = torch.tensor(data_list[0]).shape
        if len(first_item_shape) != 2:
            raise ValueError(f"Каждый элемент данных должен иметь 2 измерения (channels, seq_len), получено: {first_item_shape}")
        if first_item_shape[0] != 2: # Проверяем количество каналов
             raise ValueError(f"Ожидалось 2 канала, получено: {first_item_shape[0]}")


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Преобразуем в тензор FloatTensor, если это еще не сделано
        sample = torch.tensor(self.data_list[idx], dtype=torch.float32)
        return sample

# 2. DataLoader и параметры обучения
# Гиперпараметры
NUM_CHANNELS = 2
SEQ_LENGTH = 512  # Должно соответствовать вашей предобработке
LATENT_DIM = 32
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50 # Для примера, вам может понадобиться больше
BETA_VAE = 1.0 # Коэффициент для KL-дивергенции

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. Тренировочный цикл
def train_vae_epoch(model, dataloader, optimizer, beta, device):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0

    for batch_idx, data_batch in enumerate(dataloader):
        data_batch = data_batch.to(device) # data_batch должен быть (batch_size, NUM_CHANNELS, SEQ_LENGTH)
        
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data_batch)
        
        loss = loss_function_vae(recon_batch, data_batch, mu, log_var, beta=beta)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # Дополнительно, если хотите отслеживать компоненты потерь:
        with torch.no_grad(): # Чтобы не влиять на градиенты
            min_len = min(recon_batch.size(2), data_batch.size(2))
            recon_batch_trimmed = recon_batch[:, :, :min_len]
            data_batch_trimmed = data_batch[:, :, :min_len]
            recon_component = F.mse_loss(recon_batch_trimmed, data_batch_trimmed, reduction='sum') / data_batch.size(0)
            kld_component = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / data_batch.size(0)
            total_recon_loss += recon_component.item()
            total_kld_loss += kld_component.item()

    avg_loss = total_loss / len(dataloader) # Средняя потеря на батч
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kld_loss = total_kld_loss / len(dataloader)
    return avg_loss, avg_recon_loss, avg_kld_loss

def validate_vae_epoch(model, dataloader, beta, device):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            data_batch = data_batch.to(device)
            recon_batch, mu, log_var = model(data_batch)
            loss = loss_function_vae(recon_batch, data_batch, mu, log_var, beta=beta)
            total_loss += loss.item()

            min_len = min(recon_batch.size(2), data_batch.size(2))
            recon_batch_trimmed = recon_batch[:, :, :min_len]
            data_batch_trimmed = data_batch[:, :, :min_len]
            recon_component = F.mse_loss(recon_batch_trimmed, data_batch_trimmed, reduction='sum') / data_batch.size(0)
            kld_component = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / data_batch.size(0)
            total_recon_loss += recon_component.item()
            total_kld_loss += kld_component.item()
            
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kld_loss = total_kld_loss / len(dataloader)
    return avg_loss, avg_recon_loss, avg_kld_loss


if __name__ == '__main__':
    # --- Создание фиктивных данных для примера ---
    # В реальном сценарии здесь будут ваши предобработанные данные
    num_samples = 1000 # Общее количество кривых
    all_processed_data = []
    for _ in range(num_samples):
        # Каждая кривая - это (NUM_CHANNELS, SEQ_LENGTH)
        # Канал 0: обработанный ΔT (среднее ~0, стд ~1)
        # Канал 1: глобально нормализованная T_пробы (в диапазоне ~[0,1])
        channel1_data = np.random.randn(SEQ_LENGTH) # Пример ΔT
        channel2_data = np.random.rand(SEQ_LENGTH)  # Пример T_пробы
        stacked_sample = np.stack([channel1_data, channel2_data], axis=0)
        all_processed_data.append(stacked_sample)
    
    print(f"Сгенерировано {len(all_processed_data)} сэмплов.")
    print(f"Форма одного сэмпла: {all_processed_data[0].shape}")

    # Разделение на обучающую и валидационную выборки
    train_data, val_data = train_test_split(all_processed_data, test_size=0.2, random_state=42)

    train_dataset = DTAHeatDataset(train_data)
    val_dataset = DTAHeatDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Размер обучающего набора: {len(train_dataset)}, батчей: {len(train_loader)}")
    print(f"Размер валидационного набора: {len(val_dataset)}, батчей: {len(val_loader)}")

    # Инициализация модели и оптимизатора
    model = VAE(input_channels=NUM_CHANNELS, latent_dim=LATENT_DIM, sequence_length=SEQ_LENGTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Списки для хранения истории обучения (для графиков)
    train_losses_history = []
    val_losses_history = []
    train_recon_losses_history = []
    train_kld_losses_history = []
    val_recon_losses_history = []
    val_kld_losses_history = []


    print(f"\nНачало обучения VAE на {EPOCHS} эпох...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_recon, train_kld = train_vae_epoch(model, train_loader, optimizer, BETA_VAE, device)
        val_loss, val_recon, val_kld = validate_vae_epoch(model, val_loader, BETA_VAE, device)
        
        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)
        train_recon_losses_history.append(train_recon)
        train_kld_losses_history.append(train_kld)
        val_recon_losses_history.append(val_recon)
        val_kld_losses_history.append(val_kld)

        print(f"Эпоха {epoch}/{EPOCHS}:")
        print(f"  Train: Loss={train_loss:.4f} (Recon={train_recon:.4f}, KLD={train_kld:.4f})")
        print(f"  Valid: Loss={val_loss:.4f} (Recon={val_recon:.4f}, KLD={val_kld:.4f})")

        # Опционально: сохранение модели (например, лучшей по val_loss)
        # if epoch == 1 or val_loss < min_val_loss:
        #     min_val_loss = val_loss
        #     torch.save(model.state_dict(), 'best_vae_model.pth')
        #     print(f"  Модель сохранена как best_vae_model.pth")

    print("Обучение завершено.")

    # Визуализация процесса обучения
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(train_losses_history, label='Train Total Loss')
    plt.plot(val_losses_history, label='Validation Total Loss')
    plt.title('Общая функция потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(train_recon_losses_history, label='Train Recon Loss')
    plt.plot(val_recon_losses_history, label='Validation Recon Loss')
    plt.title('Потеря реконструкции')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(train_kld_losses_history, label='Train KLD Loss')
    plt.plot(val_kld_losses_history, label='Validation KLD Loss')
    plt.title('KL-дивергенция')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # После обучения:
    # 1. Сохранить модель: torch.save(model.state_dict(), 'final_vae_model.pth')
    # 2. Использовать model.get_latent_features(data_batch) для получения признаков
    #    для вашей основной задачи (предсказания КО и KF%).
    
    # Пример получения латентных признаков для одного батча из val_loader
    if val_loader:
        sample_val_batch = next(iter(val_loader)).to(device)
        model.eval() # Убедимся, что модель в режиме оценки
        with torch.no_grad():
            latent_features = model.get_latent_features(sample_val_batch)
        print(f"\nФорма полученных латентных признаков для одного батча: {latent_features.shape}") # (BATCH_SIZE, LATENT_DIM)
        
        # Также можно посмотреть на реконструкцию из этого батча
        recon_val, _, _ = model(sample_val_batch)
        
        # Визуализация одного примера из батча (первый сэмпл, оба канала)
        idx_to_plot = 0
        original_sample = sample_val_batch[idx_to_plot].cpu().numpy()
        reconstructed_sample = recon_val[idx_to_plot].cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(original_sample[0, :], label='Оригинал - Канал 0 (ΔT)')
        plt.plot(reconstructed_sample[0, :], label='Реконструкция - Канал 0 (ΔT)', linestyle='--')
        plt.title(f'Канал 0 (ΔT) - Сэмпл {idx_to_plot} из батча')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(original_sample[1, :], label='Оригинал - Канал 1 (T_пробы)')
        plt.plot(reconstructed_sample[1, :], label='Реконструкция - Канал 1 (T_пробы)', linestyle='--')
        plt.title(f'Канал 1 (T_пробы) - Сэмпл {idx_to_plot} из батча')
        plt.legend()
        
        plt.suptitle('Сравнение оригинала и реконструкции VAE')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Оставляем место для suptitle
        plt.show()