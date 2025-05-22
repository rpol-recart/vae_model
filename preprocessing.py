import numpy as np
import pandas as pd
from scipy.signal import savgol_filter # resample здесь не будем использовать, np.interp проще
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class GlobalTemperatureNormalizer:
    """
    Класс для глобальной Min-Max нормализации температурного канала.
    Параметры (min, max) вычисляются на обучающем наборе данных.
    """
    def __init__(self):
        self.t_min_global = None
        self.t_max_global = None

    def fit(self, temp_curves_list: list[np.ndarray]):
        """
        Вычисляет глобальный min и max по списку температурных кривых.
        Каждая кривая в списке - это np.ndarray оригинальных температур образца.
        """
        if not temp_curves_list:
            raise ValueError("Список температурных кривых не может быть пустым.")
        
        all_temps = np.concatenate([curve.flatten() for curve in temp_curves_list])
        self.t_min_global = np.min(all_temps)
        self.t_max_global = np.max(all_temps)
        print(f"Global Temp Normalizer fitted: T_min={self.t_min_global:.2f}, T_max={self.t_max_global:.2f}")

    def transform(self, temp_curve_resampled: np.ndarray) -> np.ndarray:
        if self.t_min_global is None or self.t_max_global is None:
            raise RuntimeError("Normalizer не был обучен. Вызовите fit() сначала.")
        if self.t_max_global == self.t_min_global: # Редкий случай, если все температуры одинаковые
            return np.zeros_like(temp_curve_resampled)
        
        normalized_curve = (temp_curve_resampled - self.t_min_global) / (self.t_max_global - self.t_min_global)
        # Опционально: клиппинг для значений вне [0,1] на тестовых данных
        # normalized_curve = np.clip(normalized_curve, 0, 1)
        return normalized_curve

    def fit_transform_single(self, temp_curve_resampled: np.ndarray):
        """Вспомогательный метод для случая, когда fit и transform для одной кривой (не рекомендуется для глобальной)"""
        self.t_min_global = np.min(temp_curve_resampled)
        self.t_max_global = np.max(temp_curve_resampled)
        return self.transform(temp_curve_resampled)


def preprocess_dta_curve_multichannel(
    time_arr: np.ndarray,
    temp_sample_arr: np.ndarray,         # Исходная температура образца
    temp_reference_arr: np.ndarray,
    global_temp_normalizer: GlobalTemperatureNormalizer, # Обученный нормализатор
    savgol_window_length: int = 21,
    savgol_polyorder: int = 3,
    baseline_points: int = 50,
    target_length: int = 512,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]: # Возвращает (resampled_time, stacked_channels)
    """
    Предобрабатывает одну кривую, создавая двухканальный выход для VAE.
    Канал 1: Обработанный ΔT (форма, интенсивность).
    Канал 2: Глобально нормализованная T_пробы (абсолютная температура).
    """
    if not isinstance(time_arr, np.ndarray): time_arr = np.array(time_arr)
    if not isinstance(temp_sample_arr, np.ndarray): temp_sample_arr = np.array(temp_sample_arr)
    if not isinstance(temp_reference_arr, np.ndarray): temp_reference_arr = np.array(temp_reference_arr)

    # ... (проверки входных данных и параметров SavGol как в предыдущей функции) ...
    if savgol_window_length % 2 == 0:
        savgol_window_length +=1
    if savgol_polyorder >= savgol_window_length:
        savgol_polyorder = savgol_window_length - 2 if savgol_window_length > 2 else 1
        if savgol_polyorder < 1 : savgol_polyorder = 1
    
    original_length = len(time_arr)

    # --- Канал 1: Обработанный ΔT ---
    delta_t = temp_sample_arr - temp_reference_arr

    if original_length <= savgol_window_length:
        delta_t_filtered = delta_t.copy()
    else:
        delta_t_filtered = savgol_filter(delta_t, savgol_window_length, savgol_polyorder)

    if original_length <= baseline_points:
        baseline_value_delta_t = 0 # не вычитаем, если точек мало
        delta_t_baselined = delta_t_filtered.copy()
    else:
        baseline_value_delta_t = np.mean(delta_t_filtered[:baseline_points])
        delta_t_baselined = delta_t_filtered - baseline_value_delta_t
    
    resampled_time = np.linspace(time_arr[0], time_arr[-1], target_length)
    delta_t_channel_resampled = np.interp(resampled_time, time_arr, delta_t_baselined)

    scaler_delta_t = StandardScaler() # Индивидуальный скейлер для ΔT
    delta_t_channel_normalized = scaler_delta_t.fit_transform(delta_t_channel_resampled.reshape(-1, 1)).flatten()

    # --- Канал 2: Глобально нормализованная температура образца ---
    # Ресемплируем исходную temp_sample_arr
    temp_sample_resampled = np.interp(resampled_time, time_arr, temp_sample_arr)
    
    # Нормализуем с помощью глобального нормализатора
    temp_sample_channel_normalized = global_temp_normalizer.transform(temp_sample_resampled)

    # --- Объединение каналов ---
    # VAE в PyTorch ожидает вход (batch_size, num_channels, sequence_length)
    stacked_channels = np.stack([delta_t_channel_normalized, temp_sample_channel_normalized], axis=0)
    # stacked_channels будет иметь форму (2, target_length)

    if verbose:
        plt.figure(figsize=(12, 12))

        # ... (графики для ΔT как раньше) ...
        plt.subplot(5, 1, 1)
        plt.plot(time_arr, temp_sample_arr, label='Проба (T_sample)')
        plt.plot(time_arr, temp_reference_arr, label='Эталон (T_reference)', linestyle='--')
        plt.title('Исходные температуры')
        plt.legend(); plt.grid(True)

        plt.subplot(5, 1, 2)
        plt.plot(time_arr, delta_t_filtered, label=f'ΔT после SavGol (окно {savgol_window_length})')
        if original_length > baseline_points:
            plt.plot(time_arr, delta_t_baselined, label=f'ΔT после вычитания базовой линии ({baseline_value_delta_t:.2f})', linestyle='--')
        plt.title('ΔT Канал 1: Шаги обработки')
        plt.legend(); plt.grid(True)

        plt.subplot(5, 1, 3)
        plt.plot(resampled_time, delta_t_channel_normalized, label=f'Канал 1: Финальный ΔT (норм., длина {target_length})')
        plt.title('Канал 1: Обработанный ΔT')
        plt.legend(); plt.grid(True)
        
        plt.subplot(5, 1, 4)
        plt.plot(time_arr, temp_sample_arr, label='Исходная T_пробы')
        plt.plot(resampled_time, temp_sample_resampled, label=f'T_пробы ресемплированная (длина {target_length})', linestyle='--')
        plt.title('Канал 2: Температура пробы (до глоб. нормализации)')
        plt.legend(); plt.grid(True)

        plt.subplot(5, 1, 5)
        plt.plot(resampled_time, temp_sample_channel_normalized, label=f'Канал 2: T_пробы (глобально норм., длина {target_length})')
        plt.title(f'Канал 2: T_min_g={global_temp_normalizer.t_min_global:.1f}, T_max_g={global_temp_normalizer.t_max_global:.1f}')
        plt.legend(); plt.grid(True)


        plt.tight_layout()
        plt.show()

    return resampled_time, stacked_channels

# --- Пример использования ---
if __name__ == '__main__':
    # 0. Сначала нужно "обучить" GlobalTemperatureNormalizer на всем трейн-сете
    # Предположим, у нас есть несколько примеров кривых температур
    # (В реальном сценарии это будут все ваши обучающие temp_sample_arr)
    
    # Создадим несколько синтетических наборов данных для обучения GlobalTemperatureNormalizer
    all_train_temp_curves = []
    num_train_examples_for_norm = 10 # Уменьшено для примера
    print("Генерация данных для обучения GlobalTemperatureNormalizer...")
    for i in range(num_train_examples_for_norm):
        _len = np.random.randint(4000, 7000) # Уменьшены длины для ускорения примера
        _time = np.linspace(0, _len / 100, _len)
        _initial_temp_sample = np.random.uniform(880, 980) # Более широкий диапазон для нормализатора
        _cooling_rate_sample = -0.5 - (_initial_temp_sample - 900) * 0.001
        _temp_sample = _initial_temp_sample + _cooling_rate_sample * _time
        # Добавим эффекты, чтобы температуры не были монотонными
        _peak_pos = int(_len * np.random.uniform(0.2,0.8))
        _peak_width = int(_len * 0.05)
        _temp_sample[_peak_pos : _peak_pos + _peak_width] += np.sin(np.linspace(0, np.pi, _peak_width)) * np.random.uniform(5,20)
        all_train_temp_curves.append(_temp_sample)

    global_norm = GlobalTemperatureNormalizer()
    global_norm.fit(all_train_temp_curves) # Обучаем на всем трейне

    # 1. Теперь генерируем один пример для предобработки (как будто это новый замер)
    np.random.seed(42)
    original_sequence_length = np.random.randint(4000, 8000)
    time_data = np.linspace(0, original_sequence_length / 100, original_sequence_length)
    initial_temp_sample = np.random.uniform(900, 970)
    initial_temp_ref = initial_temp_sample - np.random.uniform(-2, 2)
    cooling_rate_sample = -0.5 - (initial_temp_sample - 900) * 0.001
    cooling_rate_ref = -0.51
    temp_sample_data = initial_temp_sample + cooling_rate_sample * time_data
    temp_reference_data = initial_temp_ref + cooling_rate_ref * time_data
    peak1_pos = int(original_sequence_length * 0.3)
    peak1_width = int(original_sequence_length * 0.05)
    temp_sample_data[peak1_pos : peak1_pos + peak1_width] += np.sin(np.linspace(0, np.pi, peak1_width)) * 15
    peak2_pos = int(original_sequence_length * 0.6)
    peak2_width = int(original_sequence_length * 0.04)
    temp_sample_data[peak2_pos : peak2_pos + peak2_width] += np.sin(np.linspace(0, np.pi, peak2_width)) * 10
    temp_sample_data += np.random.normal(0, 0.5, original_sequence_length)
    temp_reference_data += np.random.normal(0, 0.3, original_sequence_length)
    
    print(f"\nОбработка примера кривой:")
    print(f"Длина исходного ряда: {original_sequence_length}")
    print(f"Начальная температура пробы: {temp_sample_data[0]:.2f}°C, пик ~{temp_sample_data[peak1_pos + peak1_width//2]:.2f}°C")


    TARGET_LEN_VAE = 512
    SAVGOL_WINDOW = 51 
    SAVGOL_POLY = 3
    BASELINE_PTS = 100

    # Проверка для очень коротких рядов
    if original_sequence_length <= SAVGOL_WINDOW:
        SAVGOL_WINDOW = max(3, original_sequence_length // 2 * 2 -1)
        SAVGOL_POLY =  max(1, SAVGOL_WINDOW -1)

    res_time, processed_channels = preprocess_dta_curve_multichannel(
        time_data,
        temp_sample_data,
        temp_reference_data,
        global_temp_normalizer=global_norm, # Передаем обученный нормализатор
        savgol_window_length=SAVGOL_WINDOW,
        savgol_polyorder=SAVGOL_POLY,
        baseline_points=BASELINE_PTS,
        target_length=TARGET_LEN_VAE,
        verbose=True
    )

    print(f"\nРазмерность обработанных каналов: {processed_channels.shape}") # Должно быть (2, TARGET_LEN_VAE)
    print(f"Канал 1 (ΔT): Mean={processed_channels[0].mean():.4f}, Std={processed_channels[0].std():.4f}")
    print(f"Канал 2 (T_пробы): Min={processed_channels[1].min():.4f}, Max={processed_channels[1].max():.4f}") # Должен быть в [0,1] или около
