import csv
import os

class DataSaver:
    def __init__(self, filename, batch_size=10):
        self.filename = filename
        self.batch_size = batch_size
        self.buffer = []

        # Перезапись файла и запись заголовка
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "label",
                "delta_x", "delta_total", "velocity", "angle", "delta_tip2",
                "dist_thumb", "dist_index", "dist_middle", "dist_ring", "dist_pinky",
                "index_ratio"
            ])

    def add_row(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.buffer)
        self.buffer = []

    def close(self):
        self.flush()
