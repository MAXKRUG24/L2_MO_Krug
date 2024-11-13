import tkinter as tk
from tkinter import messagebox, Toplevel

from Service import train, validation
from .Metric import MetricWindow


class NeuralNetworkWindow:
    def __init__(self, master):
        self.master = Toplevel(master)
        self.master.title("Нейронная Сеть")
        self.master.geometry("500x400")
        self.master.grab_set()

        # Create frames for different areas
        self.top_frame = tk.Frame(self.master)
        self.top_frame.pack(side='top', fill='x')

        self.left_frame = tk.Frame(self.master)
        self.left_frame.pack(side='left', fill='y')

        self.right_frame = tk.Frame(self.master)
        self.right_frame.pack(side='right', fill='y')

        # Input for epochs
        self.epochs_frame = tk.LabelFrame(self.top_frame, text="Ввод Эпох", padx=10, pady=10)
        self.epochs_frame.pack(side='top', fill='x', padx=10, pady=5)

        tk.Label(self.epochs_frame, text="Эпохи:").pack(side='left')
        self.epochs_entry = tk.Entry(self.epochs_frame)
        self.epochs_entry.pack(side='left')

        # Input for learning rate
        tk.Label(self.epochs_frame, text="Скорость Обучения:").pack(side='left')
        self.speed_entry = tk.Entry(self.epochs_frame)
        self.speed_entry.pack(side='left')

        # Learning options
        self.learning_frame = tk.LabelFrame(self.left_frame, text="Опции Обучения", padx=10, pady=10)
        self.learning_frame.pack(fill='x', padx=10, pady=5)

        learn_new_button = tk.Button(self.learning_frame, text="Обучить с Нуля", command=self.train_from_scratch)
        learn_new_button.pack(side='left', padx=5)

        learn_button = tk.Button(self.learning_frame, text="Дообучить", command=self.retrain)
        learn_button.pack(side='left', padx=5)

        # Validation options
        self.validation_frame = tk.LabelFrame(self.left_frame, text="Валидация", padx=10, pady=10)
        self.validation_frame.pack(fill='x', padx=10, pady=5)

        validate_button = tk.Button(self.validation_frame, text="Протестировать Модель", command=self.validate_model)
        validate_button.pack(padx=5, pady=5)

        # Metrics options
        self.metrics_frame = tk.LabelFrame(self.right_frame, text="Метрики", padx=10, pady=10)
        self.metrics_frame.pack(fill='y', padx=10, pady=5)

        graphs_button = tk.Button(self.metrics_frame, text="Графики Обучения", command=self.show_graphs_for_train)
        graphs_button.pack(padx=5, pady=5)

        graphs_button = tk.Button(self.metrics_frame, text="Графики Валидации", command=self.show_graphs_for_validation)
        graphs_button.pack(padx=5, pady=5)

    def train_from_scratch(self):
        if self.confirm_action("Вы уверены, что хотите начать обучение с нуля?"):
            epochs = self.get_epochs()
            speed = self.get_speed()
            if epochs is not None:
                print(f"Обучение с нуля на протяжении {epochs} эпох.")
                # Add your training logic here
                train(speed, epochs, True)

    def retrain(self):
        if self.confirm_action("Вы уверены, что хотите начать обучение без обнуления весов?"):
            epochs = self.get_epochs()
            speed = self.get_speed()
            if epochs and speed is not None:
                print(f"Обучение на протяжении {epochs} эпох без обнуления весов.")
                # Add your retraining logic here
                train(speed, epochs, False)

    def validate_model(self):
        print("Процесс валидации запущен...")
        epochs = self.get_epochs()
        validation(epochs)

    def get_speed(self):
        try:
            speed = float(self.speed_entry.get())
            return speed
        except ValueError:
            messagebox.showerror("Неверный ввод", "Пожалуйста, введите корректное число скорости.")
            return None

    def get_epochs(self):
        try:
            epochs = int(self.epochs_entry.get())
            return epochs
        except ValueError:
            messagebox.showerror("Неверный ввод", "Пожалуйста, введите корректное число эпох.")
            return None

    @staticmethod
    def confirm_action(message):
        return messagebox.askyesno("Подтвердите Действие", message)

    def show_graphs_for_train(self):
        MetricWindow(self.master, "train_metrics")

    def show_graphs_for_validation(self):
        MetricWindow(self.master, "validate_metrics")
