import vmiutils.pbasex._basisfn as pb

try:
    print pb.basisfn(1.0,0.908116626428, 2, 0.0, 0.424660900144, 0.0, 1.0e-7, 2)
except pb.MaxIterError:
    print "MaxIterError caught"
