import datasets


class Naive_Span_Classifier:
    """Naive classifier that memorizes whole entity spans (possibly multiple
    tokens) instead of individual tokens, and only predicts a class for a
    token sequence it has seen verbatim as a complete entity during training.
    """

    def __init__(self):
        self.memory = {}  # tuple[str, ...] span -> {entity_class: count}
        self.label_names = None
        self.max_span_len = 0

    def _extract_spans(self, tokens, tag_ids, label_names):
        spans = []
        span, entity_class = [], None
        for token, tag_id in zip(tokens, tag_ids):
            tag = label_names[tag_id]
            if tag == "O":
                if span:
                    spans.append((tuple(span), entity_class))
                span, entity_class = [], None
                continue

            prefix, tag_class = tag.split("-", 1)
            if prefix == "B" or tag_class != entity_class:
                if span:
                    spans.append((tuple(span), entity_class))
                span, entity_class = [token], tag_class
            else:  # prefix == "I" continuing a span of the same class
                span.append(token)

        if span:
            spans.append((tuple(span), entity_class))
        return spans

    def train(self, train_set: datasets.arrow_dataset.Dataset) -> dict:
        self.label_names = train_set.features["ner_tags"].feature.names

        for register in train_set:
            tokens = register.get("tokens", None)
            tags = register.get("ner_tags", None)
            if tokens is None or tags is None:
                continue

            for span, entity_class in self._extract_spans(tokens, tags, self.label_names):
                self.max_span_len = max(self.max_span_len, len(span))
                counts = self.memory.setdefault(span, {})
                counts[entity_class] = counts.get(entity_class, 0) + 1

        return self.memory

    def predict(self, test_set: datasets.arrow_dataset.Dataset) -> dict:
        label_names = self.label_names or test_set.features["ner_tags"].feature.names
        o_id = label_names.index("O")

        y = {}
        for register in test_set:
            id = register.get("id", None)
            tokens = register.get("tokens", None) or []
            n = len(tokens)
            preds = [o_id] * n

            # greedy left-to-right match, longest known span first
            i = 0
            while i < n:
                max_len = min(self.max_span_len, n - i)
                for length in range(max_len, 0, -1):
                    candidate = tuple(tokens[i:i + length])
                    counts = self.memory.get(candidate)
                    if not counts:
                        continue

                    entity_class = max(counts, key=counts.get)
                    preds[i] = label_names.index(f"B-{entity_class}")
                    for offset in range(1, length):
                        preds[i + offset] = label_names.index(f"I-{entity_class}")
                    i += length
                    break
                else:
                    i += 1

            y[id] = preds
        return y
