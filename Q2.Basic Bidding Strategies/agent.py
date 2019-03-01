class Agent:
    def __init__(self, id: int):
        self.clicks = 0
        self.id = id
    
    def addClicks(self, new_clicks: int):
        self.clicks += new_clicks