import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

marks = [ 47, 63, 71, 39, 47, 49, 43, 37, 81, 69, 38, 13, 29, 61, 49, 53, 57, 23, 58, 17, 73, 33, 29]

meanVal= np.mean(marks, axis=0)
stdVal = np.std(marks, axis=0)
print('Mean: ', np.mean(marks, axis=0))
print('Standard Deviation:', np.std(marks, axis=0))

for i in marks:
    if   (i > (meanVal+ (stdVal * 4) / 3) and i <= (meanVal+ (stdVal * 5)/3)):
        print( "Marks: ", i, "Grade : A+")
    elif (i > (meanVal+ (stdVal * 3) / 3) and i <= (meanVal+ (stdVal * 4)/3)):
        print( "Marks: ", i, "Grade : A")
    elif (i > (meanVal + (stdVal * 2) / 3) and i <= (meanVal + (stdVal * 3) / 3)):
        print("Marks: ", i, "Grade : A-")
    elif (i > (meanVal + (stdVal * 1) / 3) and i <= (meanVal + (stdVal * 2) / 3)):
        print("Marks: ", i, "Grade : B+")
    elif ( i > (meanVal) and i <= (meanVal + (stdVal * 1) / 3)):
        print("Marks: ", i, "Grade : B")
    elif (i >= (meanVal - (stdVal * 1) / 3) and i <= meanVal):
        print("Marks: ", i, "Grade : B-")
    elif (i < (meanVal - (stdVal * 1) / 3) and i >= (meanVal - (stdVal * 2) / 3)):
        print("Marks: ", i, "Grade : C+")
    elif (i < (meanVal - (stdVal * 2) / 3) and i >= (meanVal - (stdVal * 3) / 3)):
        print("Marks: ", i, "Grade : C")
    elif (i < (meanVal - ((stdVal * 3) / 3)) and i >= (meanVal - ((stdVal * 4) / 3))):
        print("Marks: ", i, "Grade : C-")
    elif (i < (meanVal - (stdVal * 4) / 3) and i >= (meanVal - (stdVal * 5) / 3)):
        print("Marks: ", i, "Grade : D")
    elif ( i < (meanVal - (stdVal * 5) / 3)):
        print("Marks: ", i, "Grade : F")