from matplotlib import pyplot as plt
import numpy as np


twoAvg=[0.8264014466546112, 0.8313743218806511, 0.8300180831826401, 0.8282097649186257, 0.8250452079566004, 0.8345388788426763, 0.8471971066907775, 0.846745027124774, 0.852622061482821, 0.8435804701627487, 0.8594032549728752, 0.8612115732368897, 0.8666365280289331, 0.8648282097649187, 0.8648282097649187, 0.8625678119349005, 0.8652802893309222, 0.8557866184448464, 0.8607594936708861, 0.8612115732368897, 0.8639240506329113, 0.8512658227848101, 0.8594032549728752, 0.8612115732368897, 0.860759493670886, 0.8589511754068716, 0.860759493670886, 0.8557866184448463, 0.8575949367088608, 0.8575949367088608, 0.8603074141048825, 0.8566907775768535, 0.8566907775768535, 0.8521699819168174]
oneAvg=[0.8264014466546112, 0.8309222423146474, 0.8318264014466547, 0.8300180831826401, 0.825497287522604, 0.8426763110307414, 0.8390596745027125,  0.8417721518987342,0.8544303797468354,0.8589511754068716, 0.8553345388788427, 0.8535262206148282, 0.8688969258589512, 0.8625678119349005, 0.8526220614828209, 0.8607594936708861, 0.8679927667269439, 0.8634719710669078, 0.8607594936708861, 0.8625678119349005, 0.8571428571428571, 0.8598553345388789, 0.8643761301989150, 0.8598553345388789, 0.8589511754068716, 0.859855334538879, 0.8598553345388789, 0.857142857142857, 0.8580470162748643, 0.8580470162748643, 0.8580470162748643, 0.8481012658227848, 0.8562386998010185, 0.8616636528028933]

y=[(2*twoAvg[i]+oneAvg[i])/3 for i in range(len(oneAvg))] 
x=range(1,len(y)+1)

standarddeviation= []
plt.plot(x,y)
#x axis show every number
plt.xticks(x)

#plot error between twoAvg and oneAvg
plt.plot(x,twoAvg,'r--')
plt.plot(x,oneAvg,'y--')

#add ledgend
plt.legend(['Average Accuracy overall','Accuracy of 2 runs','Accuracy of 1 run'])



#mark y ticks every 0.01 and the max and min values aswell
plt.yticks([i/100 for i in range(82,87)]+[0.825204,0.8672])
#plot the points in blue
plt.plot(x,y,'bo')

#fill the area between the lines



#add grid, with x=1 spacing, in grey, alternate dashes and lines
plt.grid(True, which='major', linestyle='--', color='grey')
plt.title('Average Accuracy of 3 runs vs. Number of Features')
plt.xlabel('Number of features')
plt.ylabel('Average Accuracy')
plt.show()
