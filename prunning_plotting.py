import numpy as np
import matplotlib.pyplot as plt
import random
with open("sparsity_acc_all", "r") as f:
    lines = f.readlines()
    dic = {}
    spars = []
    baseline = []
    pruned = []
    old_data = None
    for line in lines:
        if "--" not in line:
            element = line.split("_")
            mul_data = element[0]
            if old_data != mul_data:
                #create new local variables
                spars = []
                baseline = []
                pruned = []
                old_data = mul_data
            element = line.split(":")
            if "sparsity" in line:
                #print("sparsity", mul_data, float(str.rstrip(element[-1])))
                spars.append(float(str.rstrip(element[-1])))
            if "Baseline" in line:
                #print("baseline", mul_data, float(str.rstrip(element[-1])))
                baseline.append(float(str.rstrip(element[-1])))
            if "Pruned" in line:
                #print("pruned", mul_data, float(str.rstrip(element[-1])))
                pruned.append(float(str.rstrip(element[-1])))
            dic[mul_data]=[spars, baseline, pruned]
    for i, elem in enumerate(dic):
        if "TF" in elem or "MBM16" in elem or "BFLOAT" in elem: 
            print(elem)
#            plt.plot(dic[elem][0],dic[elem][1],label=elem+" baseline")
            c = "red"
            x = 2
            ls = "solid"
            if "MBM16" in elem:
                c = "green"
                x = 0
                ls = "dashdot"
            if "BFLOAT" in elem:
                c = "blue"
                x = 0
                ls = "dashdot"
            plt.axhline(y=dic[elem][1][0],alpha=0.7**x,color=c,linestyle=ls,label=elem+"baseline")
            plt.plot(dic[elem][0],dic[elem][2],marker=".",label=elem+" prunned")
            plt.legend()
            plt.savefig("example.png")
