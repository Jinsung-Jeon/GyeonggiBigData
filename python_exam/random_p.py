import random

number = random.randint(1,100)

total = 0

n = int(input('숫자를 입력하세요:'))
total +=n
answer = input('계속?')
while answer =='yes':
    total +=n
    answer = input('계속?')
    n = int(input('숫자를 입력하세요:'))
print('합계는 %d' %total)