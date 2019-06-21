#Import Calender Module to Display Calendar of Given Year and Month
import calendar
yy=int(input('Enter Year: '))
mm=int(input('Enter Month: '))
print(calendar.month(yy,mm))