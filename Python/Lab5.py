class carModel :
    def __init__(self, brand, model):
        self._brand = brand
        self._model = model

    def display_info(self):
        print(f"brand {self._brand} , model {self._model}")


car = carModel("Hyundai", "I20")

car.display_info()

