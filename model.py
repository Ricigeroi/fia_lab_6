import json
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка тренировочного и валидационного датасетов
with open('train_dataset.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('validation_dataset.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)

# Подготовка данных
start_token = '<start>'
end_token = '<end>'

# Извлечение вопросов и ответов, добавление стартового и конечного токенов к ответам
train_questions = [item['question'] for item in train_data]
train_answers = [start_token + ' ' + item['answer'] + ' ' + end_token for item in train_data]

val_questions = [item['question'] for item in val_data]
val_answers = [start_token + ' ' + item['answer'] + ' ' + end_token for item in val_data]

# Комбинирование всего текста для обучения токенизатора
all_text = train_questions + train_answers + val_questions + val_answers

# Создание токенизатора и обучение на всем тексте
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)

# Получение размера словаря
vocab_size = len(tokenizer.word_index) + 1  # Добавляем 1 из-за зарезервированного индекса 0

# Конвертация текста в последовательности
train_question_seqs = tokenizer.texts_to_sequences(train_questions)
train_answer_seqs = tokenizer.texts_to_sequences(train_answers)

val_question_seqs = tokenizer.texts_to_sequences(val_questions)
val_answer_seqs = tokenizer.texts_to_sequences(val_answers)

# Определение максимальной длины последовательностей
max_question_len = max(len(seq) for seq in train_question_seqs + val_question_seqs)
max_answer_len = max(len(seq) for seq in train_answer_seqs + val_answer_seqs)

# Паддинг последовательностей
train_question_seqs = pad_sequences(train_question_seqs, maxlen=max_question_len, padding='post')
train_answer_seqs = pad_sequences(train_answer_seqs, maxlen=max_answer_len, padding='post')

val_question_seqs = pad_sequences(val_question_seqs, maxlen=max_question_len, padding='post')
val_answer_seqs = pad_sequences(val_answer_seqs, maxlen=max_answer_len, padding='post')

# Подготовка входных и целевых последовательностей декодера для обучения
def prepare_decoder_sequences(answer_seqs):
    decoder_input_seqs = []
    decoder_target_seqs = []
    for seq in answer_seqs:
        decoder_input_seqs.append(seq[:-1])
        decoder_target_seqs.append(seq[1:])
    return decoder_input_seqs, decoder_target_seqs

train_decoder_input_seqs, train_decoder_target_seqs = prepare_decoder_sequences(train_answer_seqs)
val_decoder_input_seqs, val_decoder_target_seqs = prepare_decoder_sequences(val_answer_seqs)

# Паддинг последовательностей декодера
max_decoder_seq_len = max_answer_len - 1  # Так как мы удалили один токен
train_decoder_input_seqs = pad_sequences(train_decoder_input_seqs, maxlen=max_decoder_seq_len, padding='post')
train_decoder_target_seqs = pad_sequences(train_decoder_target_seqs, maxlen=max_decoder_seq_len, padding='post')

val_decoder_input_seqs = pad_sequences(val_decoder_input_seqs, maxlen=max_decoder_seq_len, padding='post')
val_decoder_target_seqs = pad_sequences(val_decoder_target_seqs, maxlen=max_decoder_seq_len, padding='post')

# Расширение размерности целевых последовательностей для sparse_categorical_crossentropy
train_decoder_target_seqs = np.expand_dims(train_decoder_target_seqs, -1)
val_decoder_target_seqs = np.expand_dims(val_decoder_target_seqs, -1)

# Построение модели Seq2Seq с использованием LSTM
# Энкодер
encoder_inputs = Input(shape=(max_question_len,))
encoder_embedding = Embedding(vocab_size, 256)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Декодер
decoder_inputs = Input(shape=(max_decoder_seq_len,))
decoder_embedding = Embedding(vocab_size, 256)
decoder_embedding_output = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding_output, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Определение полной модели
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Компиляция модели
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(
    [train_question_seqs, train_decoder_input_seqs],
    train_decoder_target_seqs,
    batch_size=64,
    epochs=50,
    validation_data=([val_question_seqs, val_decoder_input_seqs], val_decoder_target_seqs)
)

# Сохранение обученной модели
model.save('seq2seq_model.h5')

# Создание и сохранение моделей энкодера и декодера для инференса
# Модель энкодера
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save('seq2seq_encoder_model.h5')

# Модель декодера
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_inf = decoder_embedding(decoder_inputs)
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    decoder_embedding_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf)
decoder_model.save('seq2seq_decoder_model.h5')

# Сохранение токенизатора
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Сохранение конфигурации модели
config = {
    'max_question_len': max_question_len,
    'max_decoder_seq_len': max_decoder_seq_len,
    'vocab_size': vocab_size,
    'start_token': start_token,
    'end_token': end_token
}

with open('config.pkl', 'wb') as f:
    pickle.dump(config, f)

# Функция для декодирования последовательности (может быть использована в боте)
def decode_sequence(input_seq):
    # Получение состояний энкодера
    states_value = encoder_model.predict(input_seq)

    # Генерация пустой последовательности длины 1 для декодера
    target_seq = np.array([[tokenizer.word_index[start_token]]])

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Выбор токена с наибольшей вероятностью
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        # Добавление предсказанного слова
        decoded_sentence += ' ' + sampled_word

        # Проверка на конец последовательности или максимальную длину
        if (sampled_word == end_token or
                len(decoded_sentence.split()) > max_decoder_seq_len):
            stop_condition = True

        # Обновление целевой последовательности
        target_seq = np.array([[sampled_token_index]])

        # Обновление состояний
        states_value = [h, c]

    return decoded_sentence

# Пример использования функции декодирования
# В реальном боте вы будете использовать эту функцию для генерации ответа
def respond_to_user_input(user_input):
    # Предобработка ввода пользователя
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_seq = pad_sequences(input_seq, maxlen=max_question_len, padding='post')
    # Декодирование последовательности
    decoded_response = decode_sequence(input_seq)
    # Удаление стартовых и конечных токенов из ответа
    decoded_response = decoded_response.replace(start_token, '').replace(end_token, '').strip()
    return decoded_response

# Пример
# user_input = "Как дела?"
# response = respond_to_user_input(user_input)
# print("Бот:", response)
