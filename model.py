import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim, sequence_length): # Renamed input_dim to input_channels for clarity
        super(Encoder, self).__init__()
        # input_channels здесь - это количество каналов, для нашего случая это 2
        # sequence_length - длина временного ряда

        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=2, padding=2) # Выход: (16, sequence_length/2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)      # Выход: (32, sequence_length/4)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)      # Выход: (64, sequence_length/8)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)     # Выход: (128, sequence_length/16)
        self.bn4 = nn.BatchNorm1d(128)

        # Рассчитываем размерность после сверток для входа в линейные слои
        with torch.no_grad():
            dummy_input_shape = (1, input_channels, sequence_length) # Use input_channels here
            dummy_input = torch.randn(*dummy_input_shape)
            conv_out_shape = self._forward_conv(dummy_input).shape
            self.flattened_size = conv_out_shape[1] * conv_out_shape[2]
            
        self.fc_hidden = nn.Linear(self.flattened_size, 256) # Промежуточный полносвязный слой
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

    def _forward_conv(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, x):
        # x shape: (batch_size, input_channels, sequence_length)
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1) # Flatten
        x = F.leaky_relu(self.fc_hidden(x))
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels, sequence_length, encoder_flattened_size, encoder_conv_out_shape_spatial): # Renamed output_dim
        super(Decoder, self).__init__()
        # output_channels здесь - это количество каналов на выходе (для нас это 2)
        self.latent_dim = latent_dim
        self.encoder_flattened_size = encoder_flattened_size
        self.encoder_conv_out_channels = 128 # Должно соответствовать последнему conv слою энкодера
        self.encoder_conv_out_spatial_dim = encoder_conv_out_shape_spatial # sequence_length / 16

        self.fc_decode_hidden = nn.Linear(latent_dim, 256)
        self.fc_decode = nn.Linear(256, self.encoder_flattened_size)

        self.t_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_t1 = nn.BatchNorm1d(64)
        self.t_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_t2 = nn.BatchNorm1d(32)
        self.t_conv3 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn_t3 = nn.BatchNorm1d(16)
        # The last layer now outputs output_channels
        self.t_conv4 = nn.ConvTranspose1d(16, output_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        # Последний слой без BatchNorm.
        # Активация на выходе декодера:
        # - Канал ΔT: обычно без активации (если стандартизован) или tanh (если в [-1,1])
        # - Канал T_пробы (глобально норм. в [0,1]): можно использовать Sigmoid.
        # Если просто реконструируем 2 канала, и они имеют разные диапазоны,
        # то лучше не ставить общую активацию, а позволить MSE сделать свое дело.
        # Или применить активации по-канально ПОСЛЕ t_conv4, но это усложнит.
        # Пока оставим без явной активации в последнем слое декодера.

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        x = F.leaky_relu(self.fc_decode_hidden(z))
        x = F.leaky_relu(self.fc_decode(x))
        # Reshape to match the output shape of the encoder's convolutional part
        x = x.view(x.size(0), self.encoder_conv_out_channels, self.encoder_conv_out_spatial_dim)
        
        x = F.leaky_relu(self.bn_t1(self.t_conv1(x)))
        x = F.leaky_relu(self.bn_t2(self.t_conv2(x)))
        x = F.leaky_relu(self.bn_t3(self.t_conv3(x)))
        reconstructed_x = self.t_conv4(x) # Shape: (batch_size, output_channels, sequence_length)
        
        # Если мы хотим разные активации для разных каналов:
        # recon_channel1 = reconstructed_x[:, 0:1, :] # Без активации или Tanh
        # recon_channel2 = torch.sigmoid(reconstructed_x[:, 1:2, :]) # Sigmoid для [0,1]
        # reconstructed_x = torch.cat([recon_channel1, recon_channel2], dim=1)
        # Но для начала можно попробовать без этого, MSE может справиться.
        
        return reconstructed_x

class VAE(nn.Module):
    def __init__(self, input_channels=2, latent_dim=32, sequence_length=512): # Default input_channels=2
        super(VAE, self).__init__()
        self.input_channels = input_channels # Store for clarity, though decoder also needs it
        self.encoder = Encoder(input_channels, latent_dim, sequence_length)
        
        encoder_flattened_size = self.encoder.flattened_size
        encoder_conv_out_spatial_dim = sequence_length // (2**4) 

        # Decoder's output_channels should match encoder's input_channels if reconstructing all inputs
        self.decoder = Decoder(latent_dim, input_channels, sequence_length, encoder_flattened_size, encoder_conv_out_spatial_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x shape: (batch_size, input_channels, sequence_length)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z) # reconstruction shape: (batch_size, input_channels, sequence_length)
        return reconstruction, mu, log_var

    def get_latent_features(self, x):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu

# Функция потерь для VAE
def loss_function_vae(recon_x, x, mu, log_var, beta=1.0):
    # recon_x and x should have shape (batch_size, num_channels, sequence_length)
    
    # Убедимся, что пространственные размерности совпадают (по оси sequence_length)
    min_len = min(recon_x.size(2), x.size(2))
    recon_x_trimmed = recon_x[:, :, :min_len]
    x_trimmed = x[:, :, :min_len]

    # MSE будет вычислен по всем элементам (включая оба канала)
    # reduction='sum' суммирует потери по всем элементам в батче,
    # затем делим на размер батча для усреднения "на один пример".
    # Если хотим разный вес для каналов, нужно считать MSE по-канально и взвешивать.
    # recon_loss_ch1 = F.mse_loss(recon_x_trimmed[:, 0, :], x_trimmed[:, 0, :], reduction='sum')
    # recon_loss_ch2 = F.mse_loss(recon_x_trimmed[:, 1, :], x_trimmed[:, 1, :], reduction='sum')
    # recon_loss = (weight1 * recon_loss_ch1 + weight2 * recon_loss_ch2) / x.size(0)
    # Пока что простой MSE по всем каналам:
    recon_loss = F.mse_loss(recon_x_trimmed, x_trimmed, reduction='sum') / x.size(0)

    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kld_loss /= x.size(0)

    return recon_loss + beta * kld_loss

# --- Пример использования (с фиктивными данными) ---
if __name__ == '__main__':
    BATCH_SIZE = 4
    NUM_CHANNELS = 2 # <--- Изменение
    SEQ_LENGTH = 512
    LATENT_DIM = 32

    # Модель
    model = VAE(input_channels=NUM_CHANNELS, latent_dim=LATENT_DIM, sequence_length=SEQ_LENGTH)
    print(model)

    # Фиктивные входные данные (batch_size, num_channels, sequence_length)
    dummy_data = torch.randn(BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH)
    print(f"\nInput_data shape: {dummy_data.shape}")

    # Forward pass
    recon_data, mu, log_var = model(dummy_data)
    print(f"Reconstructed_data shape: {recon_data.shape}") # Должно быть (BATCH_SIZE, NUM_CHANNELS, SEQ_LENGTH)
    print(f"Mu shape: {mu.shape}")                         # (BATCH_SIZE, LATENT_DIM)
    print(f"Log_var shape: {log_var.shape}")               # (BATCH_SIZE, LATENT_DIM)

    # Расчет потерь
    loss = loss_function_vae(recon_data, dummy_data, mu, log_var, beta=1.0)
    print(f"Calculated loss: {loss.item()}")

    # Получение латентных признаков
    latent_vecs = model.get_latent_features(dummy_data)
    print(f"Latent features shape: {latent_vecs.shape}")   # (BATCH_SIZE, LATENT_DIM)

    # Проверка, что декодер действительно может быть создан с нужными параметрами
    # которые вычисляются в энкодере
    try:
        enc = Encoder(input_channels=NUM_CHANNELS, latent_dim=LATENT_DIM, sequence_length=SEQ_LENGTH)
        dec = Decoder(latent_dim=LATENT_DIM, 
                      output_channels=NUM_CHANNELS, 
                      sequence_length=SEQ_LENGTH, 
                      encoder_flattened_size=enc.flattened_size,
                      encoder_conv_out_shape_spatial=SEQ_LENGTH // (2**4) )
        print("\nEncoder and Decoder can be initialized independently with calculated params.")
    except Exception as e:
        print(f"\nError initializing Encoder/Decoder independently: {e}")