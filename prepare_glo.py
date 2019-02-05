
import numpy as np
egoList=[0, 107, 1684, 1912 , 3437, 348, 3980, 414, 686, 698]

gf_file = open('gf.emb', "r")
gf={}

for line in gf_file:
   a= line.rstrip('\n').split(' ')
   
   a1=a[0] 
   a2=a[1:]   
   
   vec= [float(i) for i in a2]
   
   gf.update({a1:vec})
   


f3 = open('ego.pairs' ,'w')   
f4 = open('ego.sim' ,'w')

for i in range(len(egoList)):
    distances=[]
    for j in range(len(egoList)):
        if i!= j:
            
            print(egoList[i],egoList[j])
            f3.write(str(egoList[i])+' '+str(egoList[j]))
            f3.write("\n") 
            
            d=np.dot(gf[str(egoList[i])], gf[str(egoList[j])] )
            
            distances.append((egoList[i],egoList[j], d))
            
    sorted_dist = sorted(distances, key=lambda tup: tup[2], reverse= True)
    
    print('####################################################################')
    
    for k in range(len(sorted_dist)):  
        f4.write(str(sorted_dist[k][0])+' '+str(sorted_dist[k][1])+' '+str(sorted_dist[k][2]))
        print(sorted_dist[k][0], sorted_dist[k][1], sorted_dist[k][2])
        f4.write("\n")   
     
f4.close()
f3.close()