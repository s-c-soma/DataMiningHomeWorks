Data <- c(197, 199, 234, 267,269,276,281,289, 299, 301, 339)
output <- boxplot(Data,
main = "Box Plot in R",
ylab = "Number",
col = "light grey",
border = "brown",
outline = TRUE
)
summary(Data)
boxplot.stats(Data)
output