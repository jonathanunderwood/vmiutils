import numpy
from vmiutils.simulate import NewtonSphere

bins = 256
centre=(bins/2.0, bins/2.0)

# cos^2 theta
ions1 = NewtonSphere(50.0, 8.0, [1.0, 0.0, 2.0])
vmi_img1 = ions1.vmi_image(bins, centre=centre, resample=True)
numpy.savetxt('cossq.dat', vmi_img1.image)

# sin^2 theta
ions2 = NewtonSphere(100.0, 2.0, [1.0, 0.0, -1.0])
vmi_img2 = ions2.vmi_image(bins, centre=centre, resample=True)
numpy.savetxt('sinsq.dat', vmi_img2.image)

numpy.savetxt('cossq_sinsq.dat', vmi_img1.image + vmi_img2.image)

# 5*cos^4
ions1 = NewtonSphere(50.0, 8.0, [1.0, 0.0, 20.0/7.0, 0.0, 8.0/7.0])
vmi_img1 = ions1.vmi_image(bins, centre=centre, resample=True)
numpy.savetxt('cos4.dat', vmi_img1.image)
