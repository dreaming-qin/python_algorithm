from os import read


def write_by_csv(src_filepath,des_filepath):
    src_file=open(src_filepath,'r')
    des_file=open(des_filepath,'w')
    str1='0'
    flag=True
    while str1!='':
        str1=src_file.readline()
        str1=str1.replace('\t',',')
        if flag:
            str1=','+str1
            flag=False
        des_file.write(str1)
    src_file.close()
    des_file.close()

