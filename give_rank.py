import csv

f= open('ego.sim', 'r')
r=1
sim_list=[]
for line in f:
    print(line )
    a= line.rstrip('\n').split(' ')
    #print(a)
    sim_list.append(a)


rank_matrix=[] 
for i in range(len(sim_list)):
    
    print(sim_list[i][0],sim_list[i][1], r)
    
    rank_matrix.append([sim_list[i][0],sim_list[i][1], r])
    
    r+=1
    
    if i< len(sim_list)-1 and  sim_list[i][0]!= sim_list[i+1][0]: 
        
        r=1 
    

print(rank_matrix) 
   
f1= open('rank_label.csv', 'w', newline='') 
a1 = csv.writer(f1, delimiter=',')
a1.writerows(rank_matrix)  
    
        