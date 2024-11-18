import numpy as np
from abc import ABC, abstractmethod


class MembershipFunction(ABC):
    """Base class for membership functions."""

    @abstractmethod
    def compute_membership(self, x):
        """Compute the membership degree for a given input x."""
        pass

    @abstractmethod
    def center_of_area(self, activation_value):
        """Calculate the center of area given an activation value."""
        pass


class TriangularMembershipFunction(MembershipFunction):
    def __init__(self, a, b, c):
        """
        Initialize a triangular membership function.

        :param a: Left vertex of the triangle.
        :param b: Peak of the triangle.
        :param c: Right vertex of the triangle.
        """
        self.a = a
        self.b = b
        self.c = c

    def compute_membership(self, x):
        """Compute the membership degree for a given input x."""
        if x < self.a or x > self.c:
            return 0
        elif self.a <= x <= self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b < x <= self.c:
            return (self.c - x) / (self.c - self.b)

    def center_of_area(self, activation_value):
        """Calculate the center of area given an activation value."""
        if activation_value <= 0 or activation_value > 1:
            return None  # Invalid activation value

        # Calculate the area under the triangle
        base_length = self.c - self.a
        height = activation_value

        # The center of area (centroid) for a triangle is located at 1/3 from the base to the peak
        centroid_x = (self.a + self.b + self.c) / 3

        return centroid_x


class TrapezoidalMembershipFunction(MembershipFunction):
    def __init__(self, a, b, c, d):
        """
        Initialize a trapezoidal membership function.

        :param a: Left base start.
        :param b: Left base end.
        :param c: Right base start.
        :param d: Right base end.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def compute_membership(self, x):
        """Compute the membership degree for a given input x."""
        if x < self.a or x > self.d:
            return 0
        elif self.a <= x < self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b <= x <= self.c:
            return 1
        elif self.c < x <= self.d:
            return (self.d - x) / (self.d - self.c)

    def center_of_area(self, activation_value):
        """Calculate the center of area given an activation value."""
        if activation_value <= 0 or activation_value > 1:
            return None  # Invalid activation value

        # Area under trapezoid is calculated as follows:

        # Base lengths
        base_length_left = self.b - self.a
        base_length_right = self.d - self.c

        total_area = ((base_length_left + base_length_right) / 2) * activation_value

        # Centroid calculation for trapezoid is more complex but can be approximated as follows:

        centroid_x = (
            ((self.a + self.b + (2 * (self.c + self.d))) / 4)
            * activation_value
            / total_area
        )

        return centroid_x


class GaussianMembershipFunction(MembershipFunction):
    def __init__(self, mean, sigma):
        """
        Initialize a Gaussian membership function.

        :param mean: Mean of the Gaussian.
        :param sigma: Standard deviation of the Gaussian.
        """
        self.mean = mean
        self.sigma = sigma

    def compute_membership(self, x):
        """Compute the membership degree for a given input x."""
        return np.exp(-((x - self.mean) ** 2) / (2 * (self.sigma**2)))

    def center_of_area(self, activation_value):
        """Calculate the center of area given an activation value."""

        # For Gaussian functions, we can find the approximate center based on standard properties.
        # This is more complex; in practice, we can assume it converges around its mean for significant activations.

        if activation_value <= 0 or activation_value > 1:
            return None  # Invalid activation value

        return self.mean


# Example usage:
triangular_mf = TriangularMembershipFunction(0, 5, 10)
print("Triangular Membership at 3:", triangular_mf.compute_membership(3))
print(
    "Center of Area for Triangular with activation value 0.5:",
    triangular_mf.center_of_area(0.5),
)

trapezoidal_mf = TrapezoidalMembershipFunction(0, 5, 10, 15)
print("Trapezoidal Membership at 7:", trapezoidal_mf.compute_membership(7))
print(
    "Center of Area for Trapezoidal with activation value 0.5:",
    trapezoidal_mf.center_of_area(0.5),
)

gaussian_mf = GaussianMembershipFunction(mean=5, sigma=2)
print("Gaussian Membership at 5:", gaussian_mf.compute_membership(5))
print(
    "Center of Area for Gaussian with activation value 0.5:",
    gaussian_mf.center_of_area(0.5),
)
