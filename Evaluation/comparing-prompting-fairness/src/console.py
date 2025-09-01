class DummyConsole:
    def log(self, *args, **kwargs):
        print(*args)

    def print(self, *args, **kwargs):
        print(*args)

console = DummyConsole()