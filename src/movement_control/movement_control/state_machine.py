
class StateMachine():
    def __init__(self):
        self.state = "INIT"

    def transition(self, new_state):
        self.state = new_state

