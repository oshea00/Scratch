
#         2 
#         |
#    0----1----5 6----7 
#         |    |
#         3----4 

segs = [(1,3),(1,5),(0,1),(1,2),(3,4),(4,5),(6,7)]

siz = [1,1,1,1,1,1,1,1]
cid = [0,1,2,3,4,5,6,7]
componentCount=len(cid)

def findComponent(p):
    while cid[p]!=p:
        p = cid[p]
    return p

def isConnected(p,q):
    return findComponent(p) == findComponent(q)

def union(p,q):
    global componentCount
    if not isConnected(p,q):
        idp = findComponent(p)
        idq = findComponent(q)
        if siz[idp] < siz[idq]:
            cid[idp] = idq
            siz[idq] += siz[idp]
        else:
            cid[idq] = cid[idp]
            siz[idp] += siz[idq]
        componentCount -= 1

for p,q in segs:
    union(p,q)

print(f"siz={siz}")
print(f"cid={cid}")
print(componentCount)
for p,q in segs:
    print(f"{p}->{q} {isConnected(p,q)} cid={findComponent(p)} size={siz[findComponent(p)]}")
print(f"{0}->{5} {isConnected(0,5)} cid={findComponent(5)} size={siz[findComponent(5)]}")

print(f"size of cid")
