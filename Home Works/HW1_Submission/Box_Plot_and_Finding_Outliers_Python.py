import matplotlib.pyplot as plt
data=[197, 199, 234, 267,269,276,281,289, 299, 301, 339]

fig = plt.figure()
fig.suptitle('Box plot in Python')
output = plt.boxplot(data, showcaps=True	)
print(output)
plt.show()



