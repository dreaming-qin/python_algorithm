def discretization(data,block):
    for i in range(0,len(data)):
        for j in range(0,len(block)):
            if data[i][j]<block[j][0]:
                data[i][j]=0
            elif data[i][j]>block[j][1]:
                data[i][j]=2
            else:
                data[i][j]=1
    return data
