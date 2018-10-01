def cost(data1,a1,b1) :
 sum = 0
 for l in data1 :
    sum = sum + ( a1 + b1*l[0] - l[1])**2/96
 return sum
data = [[6.1101,17.592],
[5.5277,9.1302],
[8.5186,13.662],
[7.0032,11.854],
[5.8598,6.8233],
[8.3829,11.886],
[7.4764,4.3483],
[8.5781,12],
[6.4862,6.5987],
[5.0546,3.8166],
[5.7107,3.2522],
[14.164,15.505],
[5.734,3.1551],
[8.4084,7.2258],
[5.6407,0.71618],
[5.3794,3.5129],
[6.3654,5.3048],
[5.1301,0.56077],
[6.4296,3.6518],
[7.0708,5.3893],
[6.1891,3.1386],
[20.27,21.767],
[5.4901,4.263],
[6.3261,5.1875],
[5.5649,3.0825],
[18.945,22.638],
[12.828,13.501],
[10.957,7.0467],
[13.176,14.692],
[22.203,24.147],
[5.2524,-1.22],
[6.5894,5.9966],
[9.2482,12.134],
[5.8918,1.8495],
[8.2111,6.5426],
[7.9334,4.5623],
[8.0959,4.1164],
[5.6063,3.3928],
[12.836,10.117],
[6.3534,5.4974],
[5.4069,0.55657],
[6.8825,3.9115],
[11.708,5.3854],
[5.7737,2.4406],
[7.8247,6.7318],
[7.0931,1.0463],
[5.0702,5.1337],
[5.8014,1.844],
[11.7,8.0043]
]


a = 0
b = 0
while True :
  der0 = 0
  der1 = 0
  for i in data :
    der0 = der0 + ( a + b*i[0] - i[1])/48
    der1 = der1 + ( a + b*i[0] - i[1])*i[0]/48
  
  temp1 = a - 0.0001*der0
  temp2 = b - 0.0001*der1
  print(cost(data,temp1,temp2))
  if abs(temp1 - a) < 0.0001 and abs(temp2 - b) <0.0001 :
   break
  a = temp1
  b = temp2
x = float(input("Enter population"))
sol = a + b*x
print("profit is", sol)

