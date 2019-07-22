from random import random, randint
import matplotlib.pyplot as plt
import numpy as np
import math
import lena
from scipy import misc
import matplotlib.cm as cm
import cv2
import os

dna_size = 192

def coin(p):
    i = randint(0, 1)
    return i

def random_individual(n) :
    s = []
    for i in range(n):
        c = coin(0.5)
        s.append(c)
    return s

def initial_population(N,n) : 
	P = []
	for i in range(N):
		c = random_individual(n)
		P.append(c)
	return P
 

def selection(population_fitness) :
    N = len(population_fitness)
    f = []
    P_selected = []
    for i in range(N):
        a1 = randint(0,N-1)
        a2 = randint(0,N-1)
        d_f1 = population_fitness[a1]
        d_f2 = population_fitness[a2]
        d1,f1 = d_f1
        d2,f2 = d_f2
        if abs(d1)>abs(d2):
            f0 = f2
        else:
            f0 = f1
        P_selected.append(f0)
    return P_selected 

def crossover(c) :
    c1,c2 = c
    n = len(c1)
    p = randint(0,n)
    c1_ = c1[:p] + c2[p:]
    c2_ = c2[:p] + c1[p:]
    return c1_,c2_

def mutation(k):
    n = len(k)
    p = randint(0,n-1)
    k1 = k[:]
    k1[p] = 1 - k1[p]
    return k1

def select_parents(population_fitness):
    new_population=selection(population_fitness)
    P_next=[]
    parent_pair=[]
    i=0
    while(i<len(new_population)):
        parent_pair=new_population[i],new_population[i+1]
        i=i+2
        P_next.append(parent_pair)
    return P_next


def run(iter,pop,prob_c,prob_m) :
    population = initial_population(pop,dna_size)
    generation = 1
    iter_num = iter
    best_fitness_previous = []
    global population_size
    global prob_crossover
    global prob_mutation
    population_size = pop
    prob_crossover = prob_c
    prob_mutation = prob_m
    while True :
        print("==============Current Generation: ",generation,"==============")
        fits_pops = [(lena.lena_fitness(ch),ch) for ch in population]
        best_fitness, noise = get_visulize_value(fits_pops)
        best_fitness_previous.append(best_fitness)
        lena.lena_visual(noise)

        misc.toimage((lena.GBL.lena_img+noise),cmax=255,cmin=0).save("result_created_noisy_image"+str(generation)+".png")
        misc.toimage((noise),cmax=255,cmin=0).save("result_created_noise"+str(generation)+".png")
        misc.toimage(((lena.GBL.lena_noisy-lena.GBL.lena_img)-noise),cmax=255,cmin=0).save("result_created_noise_difference"+str(generation)+".png")

        population = breed_population(fits_pops)
        if((abs(best_fitness)<abs(best_fitness_previous[generation-2]))and(generation>1)) :
            print("Got better than last generation")
        else :
            print("Got worse than last generation")
        iter_num -= 1
        generation += 1
        plt.pause(0.05)
        if(iter_num<=0): break
    return population



def breed_population(fitness_population):
    parent_pairs = select_parents(fitness_population)
    size = len(parent_pairs)
    next_population = []
    for k in range(size) :
        parents = parent_pairs[k]
        cross = random() < prob_crossover
        children = crossover(parents) if cross else parents
        for ch in children:
            mutate = random() < prob_mutation
            next_population.append(mutation(ch) if mutate else ch)
    return next_population

def get_visulize_value(p) :
    min_fitness = 100
    sum_chs = 0
    temp=[]
    for s in p:
        chs, ch = s
        sum_chs = sum_chs + chs
        if(abs(chs)<abs(min_fitness)):
            min_fitness = chs
            temp = ch
    param = lena.get_params(temp)
    original_lena, noise , lena_noise = lena.corrupt_image(lena.GBL.lena_img,param)
    p1,p2,p3 = param
    print("parameter1: ",p1)
    print("parameter2: ",p2)
    print("parameter3: ",p3) 
    print("best fitness of current generation: ",min_fitness)
    return min_fitness,noise

lena.initial()
run(100,100,0.6,0.05)

