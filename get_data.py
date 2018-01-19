import pandas as pd

def coucou(x,y):
    a =x+y
    print(a)
    return a,x,y

l,m,n=coucou(63,6)

def fact(n): #todo gérer le cas négatif
    """
    calcul le factoriel d'un nombre entier
    :param n: int
    :return: int
    """

    if n==1:
        return 1
    else:
        return n*fact(n-1)

fact(6)

def fun(x):

