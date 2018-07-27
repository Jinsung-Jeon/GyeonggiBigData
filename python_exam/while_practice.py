i = 1 
n = int(input("정수를 입력하시오:"))
fact = 1 
while i <=n:
    fact *= i 
    i += 1 
print('%d! = %d' %(n,fact))