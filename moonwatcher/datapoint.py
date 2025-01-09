class Datapoint:
    def __init__(self, id, metadata: dict = None):
        self.id = id
        self.metadata = metadata if metadata is not None else {}

    def add_metadata(self, key: str, value: any):
        self.metadata[key] = value

    def get_metadata(self, key):
        return self.metadata.get(key, None)
