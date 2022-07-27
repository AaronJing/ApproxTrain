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
        if "FP32" in elem or "Bfloat16" in elem or "AFM16" in elem: 
            print(elem)
#            plt.plot(dic[elem][0],dic[elem][1],label=elem+" baseline")
            c = "#ff7f00"
            if "AFM16" in elem:
                c = "#984ea3"
            if "Bfloat16" in elem:
                c = "#377eb8"
            if "FP32" in elem:
                plt.axhline(y=dic[elem][1][0],color="#e41a1c",linestyle="dashdot",label=elem+" baseline")
            plt.plot(dic[elem][0],dic[elem][2],color=c,marker=".",label=elem+" prunned")
            plt.legend()
    plt.xlabel("sparsity", fontweight="bold")
    plt.ylabel("test Accuracy", fontweight="bold")
    plt.savefig("example.pdf")
