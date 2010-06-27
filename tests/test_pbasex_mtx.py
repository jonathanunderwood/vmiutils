import vmiutils.pbasex as pb
import vmiutils.pbasex._basisfn as pbb

print pbb.basisfn(0.0, 0.0, 0, 0.0, 1.44, 0.0, 1.0e-7, 1.0e-7, 1000000);

a=pb.PbasexMatrix()
a.calc_matrix(1, 0, 4, 4)
print a.matrix

b=pb.PbasexMatrix()
b.calc_matrix2(1, 0, 4, 4)
print b.matrix
