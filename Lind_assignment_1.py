#Return a flattened list
def flatten(lis):
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis
    
#Return a powerset 
def powerset(list_1):
    if not list_1:
        return [[]]
    First_element_included = [ [list_1[0]] + rest for rest in powerset(list_1[1:]) ]
    First_element_not_included = powerset(list_1[1:])
    return First_element_included + First_element_not_included 

#Return all permutations   
def permu(num):
    if num:
        i , set = [],[]
        for n in num:
            if n not in set:
                removed = num[:]; removed.remove(n)
                for p in permu(removed):
                    i.append([n]+p)
            set.append(n)
        return i
    else:
        return [[]]

#Set up initial directions 
N = (0, -1)
S = (0, 1)
W = (-1, 0)
E = (1, 0) 

directions_right = {N: E, E: S, S: W, W: N} # try to turn each to the right so that old
#direction changes to new direction by 1 direction

def NumberSpiral(number_given, end_corner):
	#make sure that the inputs are correct 
    if number_given < 1 or end_corner > 4:
        return None  
    # bumps all inputs up to odd numbers   
    if (number_given % 2 == 0): 
    	number_given = number_given + 1  
    height = number_given // 2 #set up the height as the input number 
    width = number_given // 2 #set up the width as the input number 
    dir_x, dir_y = (0, -1) #initial direction for x and y 
    t = 0 #random variable to iterate 
    numspiral = [[None] * number_given for t in range(number_given)] #Creates empty grid with lists 
    count = 0 #initialize counter variable   
    while True:
        count += 1 #increment count by 1 so your lists increase by size of 1 
        numspiral[width][height] = count #width and height must equal count amount 
        # try to turn right
        new_dir_x, new_dir_y = directions_right[dir_x,dir_y] #Take old elements and turn it using new
        #elements 
        new_x = height + new_dir_x #increase the width by 1 
        new_y = width + new_dir_y #Increase the height by 1         
        if (0 <= new_x < number_given and 0 <= new_y < number_given and
            numspiral[new_y][new_x] is None): # make sure you can turn right
            height, width = new_x, new_y
            dir_x, dir_y = new_dir_x, new_dir_y      
        elif (end_corner == 2): #create it so it ends in corner 2
        	height, width = height + dir_x, width + dir_y #h and w for new direction 
        	if not (0 <= height < number_given and 0 <= width < number_given):
					return numspiral #return spiral 
        elif (end_corner == 1): #create it so it ends in corner 1 - this one does not work 
            height = height + dir_x #height for new direction (add 1) 
            width =  width + dir_y #width for new direction (add 1)
            if not (0 <= height < number_given and 0 <= width < number_given):
            	rotated_ccw = zip(*numspiral)[::-1] 
            	return rotated_ccw  #return spiral      	
        elif (end_corner == 4): #create it so it ends in corner 4
        	height, width = height + dir_x, width + dir_y #height and width for new dir
        	if not (0 <= height < number_given and 0 <= width < number_given):
            		rotated = zip(*numspiral[::-1]) #rotate 
            		return rotated         		
        elif (end_corner == 3): #create it so it ends in corner 3 
        	height, width = height + dir_x, width + dir_y #height and width for new dir
        	if not (0 <= height < number_given and 0 <= width < number_given):
        			rotation = zip(*numspiral[::1]) #rotate 
        			return rotation 	

#This is to print the numspiral out 
def print_matrix(numspiral):
    number_given = len(str(max(el for row in numspiral for el in row if el is not None)))
    fmt = "{:0%dd}" % number_given 
    for row in numspiral:
   		 print(" ".join(""*number_given if el is None else fmt.format(el) for el in row))
