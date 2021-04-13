import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
import numpy as np
import scipy as scipy
from scipy import stats
import seaborn as sns
import pandas as pd

# METHODS 
def preprocessing(name):
	'''
    This method preprocesses the data. 
    Input: .txt filename of a text file 
    Return: voltage : vector of voltage points
      		time : vector of time points
    '''        

	# open file
	fh = open(name)

	# parse data into voltage and time interval lists
	intervals =[]
	voltage=[]
	for line in fh:

		if ',' in line:
			current_voltage, current_interval = line.split()

		current_interval = current_interval[1:] # remove the comma 
		intervals.append(current_interval)
		voltage.append(current_voltage)

	# remove 1st value (non numeric character)
	intervals = intervals[1:]
	voltage = voltage[1:]

	# remove last value (if either voltage or time is missing)
	intervals = intervals[:-1]
	voltage = voltage[:-1]

	# convert to int 
	intervals = [int(x) for x in intervals]
	voltage = [int(x) for x in voltage]

	# turn intervals into timepoints 
	timepoints = [0]
	for i in range(0,len(intervals)): 
		if i == 0:
			intervals[i] = int(intervals[0])
		else:
			intervals[i] = intervals[i]+intervals[i-1]

	
	# convert analog input to voltage
	voltage_converted = [x*(5.0/1023.0) for x in voltage]

	# remove mean 
	voltage_correctmean = voltage_converted - np.mean(voltage_converted)

	# convert to np arrays 
	voltage_processed = np.array(voltage_correctmean)
	time_processed = np.array(intervals)

	

	fh.close()

	return voltage_processed,time_processed

def rms(voltage):
	'''
    This method performs root mean square calculation on the voltage vector. 
    Input: voltage : vector of voltage points
    Return: rms : vector of rms values 
      		
    '''
	rms = np.sqrt((1/len(voltage))*np.sum([x ** 2 for x in voltage]))

	return rms   

def display(name,voltage,time): 
	'''
    This method displays time vs voltage for a given sample
    Input: 
    	name: name of .txt file 
    	voltage: voltage vector 
    	time: time vector 
      		
    '''
	plt.plot(time,voltage)
	plt.title(name)
	plt.ylabel('Voltage (V)')
	plt.xlabel('Time (ms)')
	plt.ylim((-3,3))
	plt.show()



#MAIN SCRIPT

# CONTROL FILES 
ctrl = ['julia_control.txt',
'max_control.txt',
'sara_control.txt',
'stella_control.txt',
'thomas_control.txt']

dominant_0deg = ['julia_dominant_0deg.txt',
'max_dominant_0deg.txt',
'sara_dominant_0deg.txt',
'stella_dominant_0deg.txt',
'thomas_dominant_0deg.txt']

dominant_30deg = ['julia_dominant_30deg.txt',
'max_dominant_30deg.txt',
'sara_dominant_30deg.txt',
'stella_dominant_30deg.txt',
'thomas_dominant_30deg.txt']

dominant_lunge = ['julia_dominant_lunge.txt',
'max_dominant_lunge.txt',
'sara_dominant_lunge.txt',
'stella_dominant_lunge.txt',
'thomas_dominant_lunge.txt']

nondominant_0deg = ['julia_nondominant_0deg.txt',
'max_nondominant_0deg.txt',
'sara_nondominant_0deg.txt',
'stella_nondominant_0deg.txt',
'thomas_nondominant_0deg.txt']

nondominant_30deg = ['julia_nondominant_30deg.txt',
'max_nondominant_30deg.txt',
'sara_nondominant_30deg.txt',
'stella_nondominant_30deg.txt',
'thomas_nondominant_30deg.txt']

nondominant_lunge = ['julia_nondominant_lunge.txt',
'max_nondominant_lunge.txt',
'sara_nondominant_lunge.txt',
'stella_nondominant_lunge.txt',
'thomas_nondominant_lunge.txt']


# rms for control 
rms_ctrl = []
for i in ctrl: 

	voltage, time = preprocessing(i)
	rootmeansquare = rms(voltage)
	rms_ctrl.append(rootmeansquare)
	print('name:',i,'rms:',rootmeansquare)
	display(i,voltage,time)
print('ctrl rms:',np.mean(rms_ctrl))


# rms for dominant 0 degrees 
rms_dominant_0deg = []
for i in dominant_0deg: 

	voltage, time = preprocessing(i)
	rootmeansquare = rms(voltage)
	rms_dominant_0deg.append(rootmeansquare)
	print('name:',i,'rms:',rootmeansquare)
	display(i,voltage,time)
print('dominant 0deg rms:',np.mean(rms_dominant_0deg))

# rms for dominant 30 degrees
rms_dominant_30deg = []
for i in dominant_30deg: 

	voltage, time = preprocessing(i)
	rootmeansquare = rms(voltage)
	rms_dominant_30deg.append(rootmeansquare)
	print('name:',i,'rms:',rootmeansquare)
	display(i,voltage,time)
print('dominant 30deg rms:',np.mean(rms_dominant_30deg))


# nondominant 0deg
rms_nondominant_0deg = []
for i in nondominant_0deg: 

	voltage, time = preprocessing(i)
	rootmeansquare = rms(voltage)
	rms_nondominant_0deg.append(rootmeansquare)
	print('name:',i,'rms:',rootmeansquare)
	display(i,voltage,time)
print('nondominant 0deg rms:',np.mean(rms_nondominant_0deg))

#nondominant 30 degrees
rms_nondominant_30deg = []
for i in nondominant_30deg: 

	voltage, time = preprocessing(i)
	rootmeansquare = rms(voltage)
	rms_nondominant_30deg.append(rootmeansquare)
	print('name:',i,'rms:',rootmeansquare)
	display(i,voltage,time)
print('nondominant 30deg rms:',np.mean(rms_nondominant_30deg))

# dominant lunge
rms_dominant_lunge = []
for i in dominant_lunge: 

	voltage, time = preprocessing(i)
	rootmeansquare = rms(voltage)
	rms_dominant_lunge.append(rootmeansquare)
	print('name:',i,'rms:',rootmeansquare)
	display(i,voltage,time)
print('dominant lunge rms:',np.mean(rms_dominant_lunge))

# nondominant lunge
rms_nondominant_lunge = []
for i in nondominant_lunge: 

	voltage, time = preprocessing(i)
	rootmeansquare = rms(voltage)
	rms_nondominant_lunge.append(rootmeansquare)
	print('name:',i,'rms:',rootmeansquare)
	display(i,voltage,time)
print('nondominant lunge rms:',np.mean(rms_nondominant_lunge))

# RESULTS - PART 1 
d = {'ctrl': rms_ctrl, 
'dom 0deg': rms_dominant_0deg, 
'dom 30deg': rms_dominant_30deg,
'nondom 0deg': rms_nondominant_0deg,
'nondom 30deg': rms_nondominant_30deg}
df = pd.DataFrame(data=d)
print('AIM 1:')
print(df)


ax = sns.boxplot(data=df,whis=4)
ax.set(xlabel="Groups", ylabel="RMS")
plt.title("RMS during standing on one leg at 0 and 30 degree inclines (n = 5)")
#plt.xticks(rotation=10)
plt.show()


# AIM 1 : 2 WAY ANOVA 
df_aim1 = pd.DataFrame(data=[rms_ctrl, rms_dominant_0deg,rms_dominant_30deg,rms_nondominant_0deg,rms_nondominant_30deg],index=None,columns=['ctrl','dominant_0deg','dominant_30deg','nondominant_0deg','nondominant_30deg'])
print('df aim1:',df_aim1)

#RESULTS - PART 2 
d = {'dominant lunge': rms_dominant_lunge, 
'nondominant lunge': rms_nondominant_lunge}
df = pd.DataFrame(data=d)
print('AIM 2:')
print(df)
ax = sns.boxplot(data=df,whis=4)
ax.set(xlabel="Groups", ylabel="RMS")
plt.title("RMS during lunging on each side (n = 5)")
#plt.xticks(rotation=10)
plt.show()
print(df)

# AIM 2: PAIRED T TEST
t_statistic, p_value = scipy.stats.ttest_rel(rms_dominant_lunge, rms_nondominant_lunge)
print('t:',t_statistic)
print('p:',p_value)





