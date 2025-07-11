import csv
import os

class DataSaver:
    def __init__(self, filename, batch_size=10):
        self.filename = filename
        self.batch_size = batch_size
        self.buffer = []

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self._generate_header())

    def _generate_header(self):
        header = ["label"]

        # === 1. Статические признаки ===
        header.append("all_landmarks_visible")
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            header.append(f"{finger}_extended")

        # === 2. Динамика (смещение, скорость, ускорение) ===
        # Для 3-х интервалов, для запястья и кончика указательного пальца
        parts = ["wrist", "index_tip"]
        intervals = ["t0", "t1", "t2"]

        for t in intervals:
            for part in parts:
                header += [
                    f"{part}_dx_{t}", f"{part}_dy_{t}",               # смещение
                    f"{part}_vx_{t}", f"{part}_vy_{t}",               # скорость
                    f"{part}_ax_{t}", f"{part}_ay_{t}"                # ускорение
                ]

        return header

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
