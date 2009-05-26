import numpy
import math

class ChoNa:
    def __init__(self):
        pass

    def Smatrix(self, dim):
        # Note that the Cho Na i, j indices in their paper start from 1 rather
        # than 0. Here we index from zero, so the formulas are adjusted
        # accordingly.

        # Calculate the P matrix. Note that P[0][0] is zero, so iterate from 1
        # upwards.
        P = numpy.zeros((dim, dim))
        
        acos = math.acos # Avoid dictionary lookup on each iteration
        tan = math.tan

        for i in range(0, dim):
            for j in range (i, dim):
                jj = j + 1
                theta = acos (i / jj)
                P[i][j] = 0.5 * (jj * jj * theta - 
                                 i * i * tan(theta))

        self.S = numpy.zeros((dim, dim))
        
        for i in range(0, dim):
            for j in range (i, dim):
                self.S[i][j] = P[i][j]

                a = (i <= dim - 2)
                b = (j > 1)

                if a:
                    self.S[i][j] -= P[i + 1][j]

                if b: 
                    self.S[i][j] -= P[i][j - 1]

                if (a and b):
                    self.S[i][j] += P[i + 1][j - 1]


if __name__ == "__main__":
    chona = ChoNa()

    chona.Smatrix(1000)
