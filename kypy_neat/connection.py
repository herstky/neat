class Connection:
    def __init__(self, weight):
        self.input = 0
        self.weight = weight
    

    @property
    def output(self):
        return self.input * self.weight


