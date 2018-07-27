num = int(input("금액을 입력하시오:"))
m10000 =  num // 10000
num = num % 10000
m1000 =  num // 1000
num = num % 1000
m100 =  num // 100
num = num % 100
print('만원은 %d장이고 천원은 %d장 백워은 %d개이다.' %(m10000,m1000,m100))

