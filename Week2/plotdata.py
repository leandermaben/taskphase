import matplotlib.pyplot as plt
f = open("ex2data1.txt","r")
data = []
i = f.readline()
while i :
 data.append(i)
 i = f.readline()
x1 = [j[0] for j in data if j[2] == 1]
y1 = [j[1] for j in data if j[2] == 1]
plt.scatter(x1,y1,color="green",marker = "*",s=50)
x2 = [k[0] for k in data if k[2] == 0]
y2 = [k[1] for k in data if k[2] == 0]
plt.scatter(x2,y2,color="red", marker = "x", s=50)
plt.xlabel("Exam1 score")
plt.ylabel("Exam2 score")
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()
f.close()
