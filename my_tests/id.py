num = 123
mystr = 'abc'
print(id(num), num)
print(id(mystr), mystr)
num = 456
mystr = 'def'
print('修改后...')
print(id(num), num)
print(id(mystr), mystr)
