import numpy as np
from tensorflow.keras.datasets import mnist  # type: ignore
from sklearn.model_selection import train_test_split
import ssl
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Dict, Optional


# Отключение проверки SSL-сертификата.
ssl._create_default_https_context = ssl._create_unverified_context

class DataLoader:
    """Класс для загрузки и подготовки данных MNIST."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Инициализация DataLoader.

        Args:
            test_size (float): Размер тестового набора.
            random_state (int): Состояние случайности.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.x_train: np.ndarray
        self.y_train: np.ndarray
        self.x_test: np.ndarray
        self.y_test: np.ndarray
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_and_prepare_data()

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Загружает данные MNIST, разделяет на обучающий и валидационный наборы, нормализует и добавляет измерение для каналов.

        Returns:
            tuple: Кортеж из обучающих данных, меток, тестовых данных и меток.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Разделение обучающего набора на новый обучающий и валидационный.
        x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train, test_size=self.test_size, random_state=self.random_state)

        # Нормализация и добавление измерения для каналов.
        x_train_new = x_train_new.astype('float32') / 255
        x_val = x_val.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        x_train_new = x_train_new[..., np.newaxis]
        x_val = x_val[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

        # Объедение обучающего и валидационного набора для обучения на всем наборе.
        x_train_full = np.concatenate((x_train_new, x_val), axis=0)
        y_train_full = np.concatenate((y_train_new, y_val), axis=0)

        return x_train_full, y_train_full, x_test, y_test

class LeNet5:
    """Класс для создания и обучения модели LeNet-5."""

    def __init__(self, input_shape: Tuple[int, int, int] = (28, 28, 1)):
        """
        Инициализация модели LeNet-5.

        Args:
            input_shape (tuple): Форма входных данных.
        """
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self) -> Sequential:
        """
        Создает модель LeNet-5.

        Returns:
            Sequential: Модель LeNet-5.
        """
        model = Sequential([
            Conv2D(6, (5, 5), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(16, (5, 5), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(120, activation='relu'),
            Dense(84, activation='relu'),
            Dense(10, activation='softmax')
        ])
        return model

    def compile_model(self, optimizer: str = 'adam', loss: str = 'sparse_categorical_crossentropy', metrics: List[str] = ['accuracy']):
        """
        Компилирует модель LeNet-5.

        Args:
            optimizer (str): Оптимизатор.
            loss (str): Функция потерь.
            metrics (list): Список метрик.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, epochs: int = 10, batch_size: int = 128, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:

         """
        Обучает модель LeNet-5 на предоставленных обучающих данных.

        Args:
            x_train (numpy.ndarray): Набор обучающих изображений.
            y_train (numpy.ndarray): Метки классов для обучающих изображений.
            x_test (numpy.ndarray): Набор тестовых изображений.
            y_test (numpy.ndarray): Метки классов для тестовых изображений.
            epochs (int): Количество эпох.
            batch_size (int): Размер пакета.
            validation_data (tuple, optional): Валидационные данные. Defaults to None.

        Returns:
            float: Метрика F1 для предсказанных классов на тестовом наборе данных.
        """
         if validation_data is None:
            validation_data = (x_test, y_test)

         self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

         y_pred = self.model.predict(x_test)
         y_pred_class = np.argmax(y_pred, axis=1)
         f1 = f1_score(y_test, y_pred_class, average='macro')

         return f1

class ActiveLearner:
    """Класс для реализации стратегий активного обучения."""

    def __init__(self, model: Sequential, x_train_full: np.ndarray, y_train_full: np.ndarray):
        """
        Инициализация ActiveLearner.

        Args:
            model: Модель, используемая для предсказаний.
            x_train_full (numpy.ndarray): Полный набор обучающих данных.
            y_train_full (numpy.ndarray): Метки классов для полного набора обучающих данных.
        """
        self.model = model
        self.x_train_full = x_train_full
        self.y_train_full = y_train_full

    def least_confidence(self, x_unlabeled: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Реализует стратегию активного обучения Least Confidence (LC) для выбора наиболее неопределенных данных.

        Args:
            x_unlabeled (numpy.ndarray): Набор неразмеченных данных.
            num_samples (int): Количество данных для выбора.

        Returns:
            numpy.ndarray: Индексы выбранных данных.

        Примечания:
            ----------
            Сначала модель предварительно обучается на небольшом наборе данных.
            Затем предсказываются вероятности для неразмеченных данных, и выбираются те,
            для которых модель имеет наименьшую уверенность.
        """
        # Предварительное обучение.
        self.model.fit(self.x_train_full[:100], self.y_train_full[:100], epochs=5, verbose=0)

        predictions = self.model.predict(x_unlabeled, verbose=0)
        confidences = np.max(predictions, axis=1)

        indices = np.argsort(confidences)[:num_samples]

        return indices

    def bald(self, x_unlabeled: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Реализует стратегию активного обучения BALD (Bayesian Active Learning by Disagreement) для выбора данных с наибольшей неопределенностью.

        Args:
            x_unlabeled (numpy.ndarray): Набор неразмеченных данных.
            num_samples (int): Количество данных для выбора.

        Returns:
            numpy.ndarray: Индексы выбранных данных.

        Примечания:
            ----------
            Сначала модель предварительно обучается на небольшом наборе данных.
            Затем вычисляется неопределенность для неразмеченных данных как сумма отрицательных произведений вероятностей и их логарифмов.
            Выбираются данные с наибольшей неопределенностью.
        """
        # Предварительное обучение.
        self.model.fit(self.x_train_full[:100], self.y_train_full[:100], epochs=5, verbose=0)
        predictions = self.model.predict(x_unlabeled, verbose=0)

        # Добавлено 1e-9 для стабильности.
        uncertainties = np.sum(-predictions * np.log(predictions + 1e-9), axis=1)

        # Выборка с наибольшей неопределенностью.
        indices = np.argsort(uncertainties)[-num_samples:]

        return indices

# Загрузка данных.
data_loader = DataLoader()
x_train_full, y_train_full, x_test, y_test = data_loader.x_train, data_loader.y_train, data_loader.x_test, data_loader.y_test

# Инициализация модели LeNet-5.
lenet5 = LeNet5()
lenet5.compile_model()

# Обучение на всем обучающем датасете.
f1_full = lenet5.train(x_train_full, y_train_full, x_test, y_test)
print(f'F1 на тестовом наборе для модели, обученной на всем датасете: {f1_full:.4f}')

# Случайный выбор данных и обучение.
percentages: List[float] = [0.01, 0.1, 0.2]
f1_results: Dict[float, List[float]] = {p: [] for p in percentages}

for p in percentages:
    # Повтор 5 раз по ТЗ.
    for _ in range(5):
        indices = random.sample(range(len(x_train_full)), int(len(x_train_full) * p))
        x_train_sample = x_train_full[indices]
        y_train_sample = y_train_full[indices]

        lenet5_sample = LeNet5()
        lenet5_sample.compile_model()

        f1 = lenet5_sample.train(x_train_sample, y_train_sample, x_test, y_test)
        f1_results[p].append(f1)

# Расчет среднего F1 для каждой выборки.
avg_f1_values: List[float] = []
for p, f1_list in f1_results.items():
    avg_f1 = sum(f1_list) / len(f1_list)
    avg_f1_values.append(avg_f1)
    print(f'Средний F1 на тестовом наборе для {p * 100}% данных: {avg_f1:.4f}')


# Активное обучение.
f1_active_learning: Dict[float, Dict[str, List[float]]] = {p: {'LC': [], 'BALD': []} for p in percentages}


for p in percentages:
    # Повтор 5 раз по ТЗ.
    for _ in range(5):
        indices = random.sample(range(len(x_train_full)), int(len(x_train_full) * p))
        x_train_sample = x_train_full[indices]
        y_train_sample = y_train_full[indices]

        # LC.
        lenet5_lc = LeNet5()
        lenet5_lc.compile_model()

        active_learner_lc = ActiveLearner(lenet5_lc.model, x_train_full, y_train_full)
        indices_lc = active_learner_lc.least_confidence(x_train_sample, int(p * len(x_train_full)))
        x_train_lc = x_train_sample[indices_lc]
        y_train_lc = y_train_sample[indices_lc]

        f1_lc = lenet5_lc.train(x_train_lc, y_train_lc, x_test, y_test)
        f1_active_learning[p]['LC'].append(f1_lc)

        # BALD.
        lenet5_bald = LeNet5()
        lenet5_bald.compile_model()
       
        active_learner_bald = ActiveLearner(lenet5_bald.model, x_train_full, y_train_full)
        indices_bald = active_learner_bald.bald(x_train_sample, int(p * len(x_train_full)))
        x_train_bald = x_train_sample[indices_bald]
        y_train_bald = y_train_sample[indices_bald]

        f1_bald = lenet5_bald.train(x_train_bald, y_train_bald, x_test, y_test)
        f1_active_learning[p]['BALD'].append(f1_bald)


# Расчет среднего F1 для активного обучения.
avg_f1_active: Dict[float, Dict[str, float]] = {}
for p, methods in f1_active_learning.items():
    avg_f1_active[p] = {}
    for method, f1_list in methods.items():
        avg_f1 = sum(f1_list) / len(f1_list)
        avg_f1_active[p][method] = avg_f1
        print(f'Средний F1 на тестовом наборе для {p * 100}% данных с помощью {method}: {avg_f1:.4f}')

# Построение графика.
labels: List[str] = ['Полный датасет'] + [f'{p * 100}%' for p in percentages]
f1_values: List[float] = [f1_full] + avg_f1_values

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(labels, f1_values)
plt.xlabel('Метод обучения')
plt.ylabel('F1')
plt.title('Сравнение эффективности методов обучения')

plt.subplot(1, 2, 2)
for method in ['LC', 'BALD']:
    f1_values_active = [avg_f1_active[p][method] for p in percentages]
    plt.plot([p * 100 for p in percentages], f1_values_active, label=method)

plt.xlabel('Процент данных')
plt.ylabel('F1')
plt.title('Эффективность активного обучения')
plt.legend()

plt.tight_layout()
plt.show()
