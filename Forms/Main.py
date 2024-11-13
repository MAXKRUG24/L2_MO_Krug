import csv
import tkinter as tk
import numpy as np
from Service import recognition
from Service.CreateDataset import transformation_array
from .NeuroNet import NeuralNetworkWindow
import os
from dotenv import load_dotenv
from PIL import Image
import io

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Приложение для рисования")
        self.master.geometry("400x300")

        # Панель для кнопок
        button_frame = tk.Frame(self.master)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        # Кнопки
        self.recognition_button = tk.Button(button_frame, text="Распознать", command=self.picture_recognition)
        self.recognition_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(button_frame, text="Очистить", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.neural_network_button = tk.Button(button_frame, text="Нейронная сеть", command=self.open_neural_network_window)
        self.neural_network_button.pack(side=tk.LEFT, padx=5)

        # Настройка холста
        self.canvas = tk.Canvas(self.master, width=200, height=200, bg='white')
        self.canvas.pack(pady=16)

        self.canvas.bind("<B1-Motion>", self.paint)

        # Текстовое поле для вывода
        self.output_text = tk.StringVar()
        self.output_text.set("")  # Начальная пустая строка
        font = ('Helvetica', 10)  # Пример размера шрифта
        self.output_field = tk.Entry(self.master, textvariable=self.output_text, state='disabled', width=40, font=font)
        self.output_field.pack(pady=5)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')

    def picture_recognition(self):
        # Получаем постскрипт-данные с холста
        ps = self.canvas.postscript(colormode='color')

        # Используем PIL для открытия и сохранения изображения
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.convert("RGB")  # Преобразуем в RGB, если необходимо

        load_dotenv()
        const_width = int(os.getenv('const_width'))

        img = img.resize((const_width, const_width))
        # Сохраняем в BMP
        path_name = "Files/drawing"
        img.save(path_name + ".bmp", "BMP")

        img_gray = img.convert('L')

        result = []
        result.append(transformation_array(np.array(img_gray)))

        with open(path_name + ".csv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(result[0])

        result = recognition(path_name + ".csv")

        self.output_text.set(f"Результат: {result}!")

    def clear_canvas(self):
        self.canvas.delete("all")

    def open_neural_network_window(self):
        NeuralNetworkWindow(self.master)
