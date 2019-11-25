import plotly.express as px
tips = px.data.tips()
data=[197, 199, 234, 267,269,276,281,289, 299, 301, 339]
fig = px.box(data, y="total_bill")
fig.show()