import datasets
class Naive_Token_Classifier:
    def __init__(self):
        self.memory = {}

    def train(self, train_set: datasets.arrow_dataset.Dataset) -> dict:
        for register in train_set:
            for token, tag in zip(register.get("tokens", None), register.get("ner_tags", None)):
                if token is None or tag is None:
                    continue

                if tag not in self.memory:
                    self.memory[tag] = {}

                if token not in self.memory[tag]:
                    self.memory[tag][token] = 0

                self.memory[tag][token] += 1


        return self.memory


    def predict(self, test_set: datasets.arrow_dataset.Dataset) -> dict:
        y = {}
        for register in test_set:
            id = register.get("id", None)
            if id not in y.keys():
                y[id] = []

            for token in register.get("tokens", None):
                if token is None:
                    continue

                # start with nonexistent label, so OOV will count as an prediction error
                best_tag = -1
                best_count = 0
                for tag, tokens in self.memory.items():
                    count = tokens.get(token, 0)
                    if count > best_count:
                        best_count = count
                        best_tag = tag

                y[id].append(best_tag)
        return y

