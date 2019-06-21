#Using the power of datetime 
from  datetime import date

def cal_calender():
    f_date=date(2014, 7, 2)
    l_date=date(2014, 7, 11)
    #subtract two date objects
    res_date=l_date - f_date
    print(res_date.days,"days")

cal_calender()