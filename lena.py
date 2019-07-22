import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import random, randint
from scipy import misc
import os
import cv2

size = [512,512]

#fn = "/Users/zihao/lopez/ga_project_2018.09.28/lena.png"

def gene2val(gene,ab) :
	a, b = ab
	n = len(gene)
	M = 2 ** n -1
	#print("gene: ",gene)
	str_gene=""
	for each in gene:
		str_gene=str(each)+str_gene
	#str_gene="".join(gene)
	r = int(str_gene, 2)
	x = a + r / M * (b-a)
	return x

class GBL:
	lena_img = []
	lena_noisy = []
	lena_normalize = []

def imread(fn) :
	#return np.array(Image.open(fn).convert("L"), dtype=np.float)
	return cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float)

def imwrite(fn, im_array) :
	#Image.fromarray(np.uint8(im_array)).save(fn)
	cv2.imwrite(fn, im_array)

def corrupt_image(im_in, noise_params) : 
	#read image
	signal = im_in
	if type(im_in) == type("") :
		signal = imread(im_in)
	#noise
	noise = make_noise(size, noise_params)
	#corrupt it
	signal_noisy = signal + noise
	#voila
	return signal, noise, signal_noisy



#def fitness(lena_noisy, lena_noisy_ga) :
def im_diff_val(a, b) :
	d = np.abs(a-b)
	f = np.sum(d)
	if 1 :
		n, m = a.shape
		M = n*m*255
	return f / M # [0,1]

def make_noise(im_size, noise_params) : 
	#n, m = im_size
	
	#NoiseAmp, NoiseFreqRow, NoiseFreqCol = noise_params
	#rows, cols = im_size
	#N = np.zeros(im_size)
	#row, col = np.arange(rows), np.arange(cols).reshape((cols,rows))
	#N = NoiseAmp*np.sin( 2*np.pi*NoiseFreqRow*row + 2*np.pi*NoiseFreqCol*col )
	
	#return N
	NoiseAmp,NoiseFreqRow, NoiseFreqCol = noise_params
	h, w = size
	zero_offset = 0
	zero_offset = 1
	y = np.arange(h) + zero_offset
	x = np.arange(w) + zero_offset
	col, row = np.meshgrid(x, y, sparse=True)
	noise = NoiseAmp * np.sin(2*np.pi * NoiseFreqRow * row + 2*np.pi * NoiseFreqCol * col)
	return noise

#def get_normalized_params(s,max_value) :
	#size = len(s)-1
	#v0 = 0.0
	#v = 0.0
	#for each in s :
	#	v0 += each*(math.pow(2,size))
	#	size -= 1
	#v = v0/(math.pow(2,64)) * max_value
	#return v
	
def get_params(gene) :
	#print("gene_params: ", gene)
	param1 = gene[:64]
	p1 = gene2val(param1,[0.0,30.0])
	param2 = gene[64:128]
	p2 = gene2val(param2,[0.0,0.01])
	param3 = gene[128:]
	p3 = gene2val(param3,[0.0,0.01])
	p = p1,p2,p3
	return p	
'''
def draw(p,ymax) :
	plt.plot(range(len(p)),p)
	plt.axis([1,ymax,0,-30])
	plt.show()
'''
def lena_visual(noise) :

	plt.clf()
	plt.subplot(326)
	plt.title('difference')
	#print(noise)
	plt.imshow((GBL.lena_noisy-GBL.lena_img)-noise,cmap=cm.gray)
	#misc.toimage(((GBL.lena_noisy-GBL.lena_img)-noise),cmax=255,cmin=0).save("diff.png")
	#plt.imshow((noise)-noise,cmap=cm.gray)

	plt.subplot(325)
	plt.title('noise')
	plt.imshow(noise,cmap=plt.cm.gray)

	plt.subplot(321)
	plt.title('original image')
	plt.imshow(GBL.lena_img,cmap=plt.cm.gray)

	plt.subplot(322)
	plt.title('original noisy image')
	plt.imshow(GBL.lena_noisy,cmap=plt.cm.gray)

	plt.subplot(323)
	plt.title('original noise')
	plt.imshow(GBL.lena_noisy-GBL.lena_img,cmap=plt.cm.gray)

	plt.subplot(324)
	plt.title('noisy image')
	plt.imshow(GBL.lena_img+noise,cmap=plt.cm.gray)

def initial() :
	GBL.lena_img = imread("lena.png")
	GBL.lena_noisy = imread("lena.png_noisy_NA_XXX_NFRow_XXX_NFCol_XXX.png")
	GBL.lena_normalize = np.prod(GBL.lena_img.shape[:2])

def lena_fitness(gene) :
	noise_params = get_params(gene)
	lena, noise , lena_noisy = corrupt_image(GBL.lena_img,noise_params)
	#noise_diff = np.sum(np.abs(GBL.lena_noisy-lena_noisy)) / GBL.lena_normalize
	noise_diff = im_diff_val(GBL.lena_noisy,lena_noisy)
	f = - noise_diff
	return f



initial()

#print(GBL.lena_img)

