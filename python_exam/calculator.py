a = float(input("a를 입력하시오:"))
b = float(input("b를 입력하시오:"))
c = input("연산자를 입력하시오(+,-,*,/):")

if c == "+" :
    print(a,c,b,"=", a+b)
elif c =="-" :
    print(a,c,b,"=", a-b)
elif c =="*":
    print(a,c,b,"=", a*b)
elif c =="/":
    if b ==0:
        print("안돼요")
    else:
        print(a,c,b,"=",float(a%b))

    80000/