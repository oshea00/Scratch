def sum(a):
    if len(a) == 0:
        return 0
    return a.pop() + sum(a)

print(sum([1,2,3,12,5,6,7,10]))


