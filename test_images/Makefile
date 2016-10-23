all: 128x128x64x4.mtx im_example.pbfit im_example.pdf

.PHONY: all

128x128x64x4.mtx:
	pbmatrix calc --no-oddl --sigma 1.5 --no-oddl 128 128 64 4 $@

im_example.pbfit: im_example.asc 128x128x64x4.mtx
	pbfit \
	--swapxy \
	--centre 220.0 220.0 \
	--lmax 4 \
	--no-oddl \
	--method 'least_squares' \
	$^ $@

im_example.pdf: im_example.pbfit
	pbfitplt $^ $@
