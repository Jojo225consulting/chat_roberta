class CombineLabels:
    def __init__(self, tokenized_data):
        # map avec la fonction statique
        self.dataset_with_labels = tokenized_data.map(self.combine_labels)

    @staticmethod
    def combine_labels(example):
        example["labels"] = [
            float(example["O"]),
            float(example["C"]),
            float(example["E"]),
            float(example["A"]),
            float(example["N"]),
        ]
        return example
