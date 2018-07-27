ss = 'python을 열심히 공부 중'
ss.split()

ss = '하나:둘:셋'
ss.split(':')

ss = input("날짜(연/월/일) 입력 ==>")
ssList = ss.split('/')

print("입력한 날짜의 10년 후 ==>",end='')
print(str(int(ssList[0])+10)+"년",end='')
print(ssList[1]+"월",end='')
print(ssList[2]+"일")

#함수명에 대입
before=['2019','12','31']
after=list(map(int,before))
after

#문자열
a= 1234
b=12.34
c="hello"
d="123Abc"
isalnum(a)

#리스트 데이터삽입(append(), insert()),삭제(remove(항목값),del리스트[],pop),요소찾기(index(),count())
#정렬(sort(),sorted())
heroes=["아이언맨","토르","헐크","스칼렛위치"]
last_hero=heroes.pop()
print(last_hero)

print(heroes.index("헐크"))

for hero in heroes:
    print(hero)

heroes.sort()
print(heroes)

numbers=[9,6,7,3,8,4,5,3,2]
numbers.sort()
print(numbers)
new_list=sorted(numbers)
print(new_list)

a=[1,3,-1,3]
b=a
a
b
a.append(10)
a
b
#list를 복사할때 따로 복사해줘야해 
b = a[:]
id(b)
c = a.copy()
c

a = [1,2,3]

fruits = ['orange','apple','pear','banana','kiwi','apple','banana']
fruits.count('apple')

fruits.index('banana')
fruits.index('banana',4) #find next banana starting a position 4
nest = [[1,2,3],[4,5,6],[7,8,9]]
nest
nest[0]
nest[0][0]

import turtle
def draw_olympic_symbol():
    positions = [[0, 0, "blue"], [-120, 0, "purple"], [60,60, "red"], [-60, 60, "yellow"], [-180, 60, "green"]]
    for x, y, c in positions:
        t.penup()
        t.goto(x, y)
        t.pendown()
        t.color(c, c)
        t.begin_fill()
        t.circle(30)
        t.end_fill()
t = turtle.Turtle()
draw_olympic_symbol()

dict = {'Name': '홍길동', 'Age': 7, 'Class': '초급'}
dict['Name']
dict['Age']


words = {"one":"하나","two":"둘","three":"셋","four":"넷"}
word = input("영어를 입력하시오:");
print(words[word])

engDict = {}
engDict["apple"] = "사과"
engDict["victory"]="승리"
engDict["love"]="사랑"

word= inpurt('영어단어를 입력:')
sen = endDict[word]
print(sen)

#[] list , {} dict(),

if score >= 60:
    print("합격입니다.")
else:
    print("불합격입니다.")

if score > 90:
    print("합격입니다.")
    print("장학급받을수있습니다.")

score = int(input("성적을 입력하시오:"))
if score >= 60:
    print("합격입니다.")
else
    print("불합격입니다.")

num = int(input("정수를 입력하시오:"))
if num % 2 ==0:
    print("짝수입니다.")
else:
    print("홀수입니다.")

    
money = 57800
num = int(input("금액을 입력하시오:"))
m10000 =  num // 10000
num = num % 10000
m1000 =  num // 1000
num = num % 1000
m100 =  num // 100
num = num % 100
print('만원은 %d장이고 천원은 %d장 백워은 %d개이다.' %(m10000,m1000,m100))


          
