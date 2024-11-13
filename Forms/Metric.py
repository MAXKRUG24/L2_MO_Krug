import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
import matplotlib.backends.backend_tkagg


class MetricWindow:
    def __init__(self, master, file_named):
        self.master = master
        file_path = f'Files/metrics/{file_named}.csv'
        data = pd.read_csv(file_path, sep=';', header=None,
                           names=['loss', 'accuracy', 'precision', 'recall'])

        self.data = data

        # Immediately display the plots upon initializing the class
        self.show_plots()

    def show_plots(self):
        # Create a new window for the plots
        plot_window = Tk()
        plot_window.title("Metrics Graphs")

        # Create a figure for the plots
        fig, axs = plt.subplots(4, 1, figsize=(8, 12))  # 4 rows, 1 column

        # Loss
        axs[0].plot(self.data.index, self.data['loss'], label='Loss', color='blue')
        axs[0].set_title('Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Value')
        axs[0].grid()

        # Accuracy
        axs[1].plot(self.data.index, self.data['accuracy'], label='Accuracy', color='green')
        axs[1].set_title('Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Value')
        axs[1].grid()

        # Precision
        axs[2].plot(self.data.index, self.data['precision'], label='Precision', color='orange')
        axs[2].set_title('Precision')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Value')
        axs[2].grid()

        # Recall
        axs[3].plot(self.data.index, self.data['recall'], label='Recall', color='red')
        axs[3].set_title('Recall')
        axs[3].set_xlabel('Epochs')
        axs[3].set_ylabel('Value')
        axs[3].grid()

        # Adjust layout
        plt.tight_layout()

        # Embed the plots in Tkinter
        canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=plot_window)
        canvas.get_tk_widget().pack()
        canvas.draw()

        plot_window.mainloop()


def main():
    # Read data from the CSV file
    file_name = 'data.csv'  # Specify the name of your CSV file
    data = pd.read_csv(file_name, sep=',', header=None, names=['loss', 'accuracy', 'precision', 'recall'])

    # Create the main Tkinter window and display plots
    root = Tk()
    root.title("Metrics from CSV")

    # Instantiate the MetricWindow class
    app = MetricWindow(root, data)


if __name__ == '__main__':
    main()
