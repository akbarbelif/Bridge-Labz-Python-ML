def add_set():
    color_set = set()
    num = int(input('Input Total Set: '))
    for i in range(1, num+1):
        val = input(i+'Enter Color Name: ')
        color_set.add(val)
    print(color_set)


add_set()
