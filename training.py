import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle

from dataset import get_haiku_dataset


class HaikuGenerator:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.vocab_size = 0
        self.max_sequence_length = 0
        
    def prepare_data(self, haikus):
        """
        Подготовка данных для обучения
        haikus: список хокку в формате [строка1, строка2, строка3]
        """
        # Объединяем все строки хокку в один текст для создания словаря
        all_text = []
        training_sequences = []
        
        for haiku in haikus:
            # Каждое хокку представляем как одну строку с разделителями
            full_haiku = haiku[0] + " <line> " + haiku[1] + " <line> " + haiku[2] + " <end>"
            all_text.append(full_haiku)
        
        # Создаем токенизатор
        self.tokenizer = Tokenizer(char_level=False, 
                                 filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                                 oov_token="<unk>")
        self.tokenizer.fit_on_texts(all_text)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        # Подготавливаем входные и выходные последовательности
        input_sequences = []
        target_sequences = []
        
        for haiku in haikus:
            # Создаем полную последовательность
            full_sequence = self.tokenizer.texts_to_sequences([
                haiku[0] + " <line> " + haiku[1] + " <line> " + haiku[2] + " <end>"
            ])[0]
            
            # Находим позицию первого <line> токена
            line_token_id = self.tokenizer.word_index.get("<line>", 0)
            first_line_end = -1
            
            for i, token_id in enumerate(full_sequence):
                if token_id == line_token_id:
                    first_line_end = i
                    break
            
            if first_line_end == -1:
                continue
                
            # Создаем обучающие пары: от начала до позиции i -> следующий токен
            for i in range(first_line_end + 1, len(full_sequence)):
                input_seq = full_sequence[:i]
                target_word = full_sequence[i]
                
                input_sequences.append(input_seq)
                target_sequences.append(target_word)
        
        # Паддинг последовательностей
        if input_sequences:
            self.max_sequence_length = max(len(seq) for seq in input_sequences)
            input_sequences = pad_sequences(input_sequences, maxlen=self.max_sequence_length, padding='pre')
            
            # Преобразование целей в категориальный формат
            target_sequences = to_categorical(target_sequences, num_classes=self.vocab_size)
            
            return input_sequences, target_sequences
        else:
            raise ValueError("Не удалось создать обучающие последовательности")
    
    def create_model(self):
        """Создание модели LSTM"""
        self.model = Sequential([
            Embedding(self.vocab_size, 64, input_length=self.max_sequence_length),
            LSTM(100, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            LSTM(100, dropout=0.3, recurrent_dropout=0.3),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, haikus, epochs=100, batch_size=32):
        """Обучение модели"""
        print("Подготовка данных...")
        X, y = self.prepare_data(haikus)
        
        print(f"Создание модели с размером словаря: {self.vocab_size}")
        self.create_model()
        
        print("Начало обучения...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.2
        )
        
        return history
    
    def generate_haiku_completion(self, first_line, max_length=20, temperature=0.7):
        """
        Генерация завершения хокку по первой строке
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Модель должна быть обучена перед генерацией")
        
        # Создаем обратный словарь для токенов
        reverse_word_map = {v: k for k, v in self.tokenizer.word_index.items()}
        
        # Токенизация первой строки
        seed_text = first_line + " <line>"
        generated_words = []
        current_text = seed_text
        
        for _ in range(max_length):
            # Токенизация текущей последовательности
            token_list = self.tokenizer.texts_to_sequences([current_text])[0]
            
            # Паддинг
            if len(token_list) > self.max_sequence_length:
                token_list = token_list[-self.max_sequence_length:]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_length, padding='pre')
            
            # Предсказание следующего слова
            predicted_probs = self.model.predict(token_list, verbose=0)[0]
            
            # Применение температуры для разнообразия
            predicted_probs = np.array(predicted_probs)
            predicted_probs = predicted_probs ** (1/temperature)
            predicted_probs = predicted_probs / np.sum(predicted_probs)
            
            # Выбор следующего слова (исключаем токен 0 - padding)
            predicted_id = np.random.choice(range(1, len(predicted_probs)), p=predicted_probs[1:]/np.sum(predicted_probs[1:]))
            
            # Преобразование обратно в слово
            output_word = reverse_word_map.get(predicted_id, "<unk>")
            
            # Проверка на завершение
            if output_word == "<end>":
                break
            
            if output_word == "<unk>":
                continue
                
            generated_words.append(output_word)
            current_text += " " + output_word
        
        # Форматирование результата
        result_text = " ".join(generated_words)
        
        # Разделяем строки хокку
        lines = result_text.split(" <line> ")
        formatted_lines = []
        
        for line in lines:
            # Очищаем от лишних токенов
            clean_line = line.replace("<line>", "").replace("<end>", "").strip()
            if clean_line:
                formatted_lines.append(clean_line)
        
        # Ограничиваем до 2 строк (вторая и третья строки хокку)
        if len(formatted_lines) > 2:
            formatted_lines = formatted_lines[:2]
        
        return "\n".join(formatted_lines)
    
    def save_model(self, model_path="haiku_model.h5", tokenizer_path="tokenizer.pkl"):
        """Сохранение модели и токенизатора"""
        if self.model:
            self.model.save(model_path)
        
        if self.tokenizer:
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
    
    def load_model(self, model_path="haiku_model.h5", tokenizer_path="tokenizer.pkl"):
        """Загрузка модели и токенизатора"""
        self.model = tf.keras.models.load_model(model_path)
        
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        self.vocab_size = len(self.tokenizer.word_index) + 1

# Пример использования
def main():
    # Создание и обучение модели
    generator = HaikuGenerator()
    haiku_data = get_haiku_dataset()
    
    print("Начинаем обучение модели...")
    history = generator.train(haiku_data, epochs=200, batch_size=32)
    
    # Тестирование генерации
    print("\nГенерация хокку:")
    first_lines = [
        "Утренний туман",
        "Цветущая сакура",
        "Зимний вечер тих"
    ]
    
    for first_line in first_lines:
        print(f"\nПервая строка: {first_line}")
        completion = generator.generate_haiku_completion(first_line)
        print("Завершение:")
        print(completion)
        print("-" * 30)
    
    # Сохранение модели
    generator.save_model()
    print("\nМодель сохранена!")

if __name__ == "__main__":
    main()