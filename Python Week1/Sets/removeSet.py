def remove_set():
    res_set=set();
    emp_name=input('Enter Employee Name :')
    res_set=emp_name.split(",")
    print('Employee Name',res_set)
    Join=input('Do you wanna Remove any Employee?\n ')
    if Join in ['yes','Yes','y']:
        rv=input('Enter the name of employee to Remove :')
        res_set.remove(rv)
        print(res_set)
    else:
        print(res_set)

remove_set()