class KeyboardHandler:
    """
    Substitute for pygame.key.get_pressed() as this list seems to drop inputs
    """
    def __init__(self):
        self.pressed = set()

    def get_key_pressed(self, key):
        return key in self.pressed

    def key_pressed(self, key):
        self.pressed.add(key)

    def key_released(self, key):
        self.pressed.remove(key)
