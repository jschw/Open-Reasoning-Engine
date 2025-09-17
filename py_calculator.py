import math

class PiCalculator:
    """
    Calculates Pi to a specified precision using the Leibniz formula.
    """

    def __init__(self, precision):
        """
        Initializes the PiCalculator with the desired precision.

        Args:
            precision (float): The desired precision of Pi.
        """
        self.precision = precision
        self.pi_approx = 0.0
        self.term = 1.0
        self.sign = 1.0
        self.n = 0

    def calculate_pi(self):
        """
        Calculates Pi to the specified precision using the Leibniz formula.
        """
        while True:
            self.pi_approx += self.term
            self.n += 1
            self.term = -self.term / (2 * self.n - 1)
            print(self.term)
            if abs(self.term) < self.precision:
                break

        return self.pi_approx * 4

# Example usage:
if __name__ == "__main__":
    precision = 0.000000001
    calculator = PiCalculator(precision)
    pi = calculator.calculate_pi()
    print(f"Pi to {precision} precision: {pi}")
    print(f"Math.pi: {math.pi}")