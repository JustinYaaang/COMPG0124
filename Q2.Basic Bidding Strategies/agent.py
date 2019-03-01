class Agent:
    def __init__(self):
        self.clicks = 0
    
    def updateClicks(self, new_clicks: int):
        self.clicks += new_clicks