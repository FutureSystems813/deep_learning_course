class Car:
    def __init__(self, name, horse_power, price):
        self.name = name
        self.horse_power = horse_power
        self.price = price

    def power_price_ratio(self):
        ratio = self.horse_power / self.price
        return ratio

    def __repr__(self):
        return "Car: {0} with {1} hp and a price of {2}".format(self.name, self.horse_power, self.price)

if __name__ == "__main__":
    audi_a1 = Car("Audi A1", 200, 38000)
    ratio = audi_a1.power_price_ratio()
    print("Ratio: ", ratio)
    print(audi_a1)