import re

def basify(wd):
    return wd

def clean(wd):
    return wd

def remove_str(wd):
    list_of_str = list('1234567890(),.;?!')
    for i in list_of_str:
        wd = re.sub(i,'',wd)
    return wd
    

