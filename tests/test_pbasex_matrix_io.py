import sys
import vmiutils.pbasex as pb

file = "test_pbasex_matrix_io.dat"

M = pb.PbasexMatrix()
M.calc_matrix(1, 0, 4, 4)
M.dump(file)

N = pb.PbasexMatrix()
N.load(file)

print N.kmax
print N.matrix

