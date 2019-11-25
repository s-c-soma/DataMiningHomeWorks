# Load libraries
library(tidyverse)
library(corrplot)
library(gridExtra)
library(GGally)
library(knitr)
library(klaR)


# Read the stats
wines <- read.csv("C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/HW4_Submission/wine.csv")
data_ <- read.csv("C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/HW4_Submission/wine.csv")

# Remove the Type column
wines <- wines[, -14]

# Histogram for each Attribute
wines %>%
  gather(Attributes, value, 1:13) %>%
  ggplot(aes(x=value, fill=Attributes)) +
  geom_histogram(colour="black", show.legend=FALSE) +
  facet_wrap(~Attributes, scales="free_x") +
  labs(x="Values", y="Frequency",
       title="Wines Attributes - Histograms") +
  theme_bw()

# Density plot for each Attribute
wines %>%
  gather(Attributes, value, 1:13) %>%
  ggplot(aes(x=value, fill=Attributes)) +
  geom_density(colour="black", alpha=0.5, show.legend=FALSE) +
  facet_wrap(~Attributes, scales="free_x") +
  labs(x="Values", y="Density",
       title="Wines Attributes - Density plots") +
  theme_bw()


# Boxplot for each Attribute  
wines %>%
  gather(Attributes, values, c(1:4, 6:12)) %>%
  ggplot(aes(x=reorder(Attributes, values, FUN=median), y=values, fill=Attributes)) +
  geom_boxplot(show.legend=FALSE) +
  labs(title="Wines Attributes - Boxplots") +
  theme_bw() +
  theme(axis.title.y=element_blank(),
        axis.title.x=element_blank()) +
  ylim(0, 35) +
  coord_flip()

# Correlation matrix 
corrplot(cor(wines), type="upper", method="ellipse", tl.cex=0.9)

# Relationship between Phenols and Flavanoids
ggplot(wines, aes(x=Total_phenols, y=Flavanoids)) +
  geom_point() +
  geom_smooth(method="lm", se=FALSE) +
  labs(title="Wines Attributes",
       subtitle="Relationship between Phenols and Flavanoids") +
  theme_bw()

# Normalization
winesNorm <- as.data.frame(scale(wines))

# Original data
p1 <- ggplot(wines, aes(x=Alcohol, y=Malic_Acid)) +
  geom_point() +
  labs(title="Original data") +
  theme_bw()

# Normalized data 
p2 <- ggplot(winesNorm, aes(x=Alcohol, y=Malic_Acid)) +
  geom_point() +
  labs(title="Normalized data") +
  theme_bw()

# Subplot
grid.arrange(p1, p2, ncol=2)

# Execution of k-means with k=2
set.seed(1234)
wines_kmean2 <- kmeans(winesNorm, centers=2)

# Cluster to which each point is allocated
wines_kmean2 $cluster

# Cluster centers
wines_k2$centers

# Cluster size
wines_kmean2 $size

# Between-cluster sum of squares
wines_kmean2 $betweenss

# Within-cluster sum of squares
wines_kmean2 $withinss

# Total within-cluster sum of squares 
wines_kmean2 $tot.withinss

# Total sum of squares
wines_kmean2 $totss

#######Code to choose optimal number of cluster-for k-mean#####################
bss <- numeric()
wss <- numeric()

# Run the algorithm for different values of k 
set.seed(1234)

for(i in 1:10){
  
  # For each k, calculate betweenss and tot.withinss
  bss[i] <- kmeans(winesNorm, centers=i)$betweenss
  wss[i] <- kmeans(winesNorm, centers=i)$withinss
  
}

# Between-cluster sum of squares vs Choice of k
p3 <- qplot(1:10, bss, geom=c("point", "line"), 
            xlab="Number of clusters", ylab="Between-cluster sum of squares") +
  scale_x_continuous(breaks=seq(0, 10, 1)) +
  theme_bw()

# Total within-cluster sum of squares vs Choice of k
p4 <- qplot(1:10, wss, geom=c("point", "line"),
            xlab="Number of clusters", ylab="Total within-cluster sum of squares") +
  scale_x_continuous(breaks=seq(0, 10, 1)) +
  theme_bw()

# Subplot
grid.arrange(p3, p4, ncol=2)
#####################################################################
# Execution of k-means with k=3
set.seed(1234)

wines_kmean3 <- kmeans(winesNorm, centers=3)

# Cluster to which each point is allocated
wines_kmean3 $cluster

# Cluster centers
wines_kmean3$centers

# Cluster size
wines_kmean3 $size

# Between-cluster sum of squares
wines_kmean3 $betweenss

# Within-cluster sum of squares
wines_kmean3 $withinss

# Total within-cluster sum of squares 
wines_kmean3 $tot.withinss

# Total sum of squares
wines_kmean3 $totss

# Iterayions
wines_kmean3 $iter

# Mean values of each cluster
aggregate(wines, by=list(wines_kmean3$cluster), mean)

# Clustering 
ggpairs(cbind(wines, Cluster=as.factor(wines_kmean3$cluster)),
        columns=1:6, aes(colour=Cluster, alpha=0.5),
        lower=list(continuous="points"),
        upper=list(continuous="blank"),
        axisLabels="none", switch="both") +
  theme_bw()

#checking
table(data_$Class,wines_kmean3$cluster)
##################################################################################################
#######Code to choose optimal number of cluster-for k-mode#####################
bss <- numeric()
wss <- numeric()

# Run the algorithm for different values of k 
set.seed(1234)

for(i in 1:10){
  
  # For each k, calculate betweenss and tot.withinss
  wss[i] <- kmodes(winesNorm, i)$withindiff
  
}
wss[1] <- kmodes(winesNorm, 1)$withindiff
wss[2] <- kmodes(winesNorm, 2)$withindiff
wss[3] <- kmodes(winesNorm, 3)$withindiff
wss[4] <- kmodes(winesNorm, 4)$withindiff
wss[5] <- kmodes(winesNorm, 5)$withindiff
wss[6] <- kmodes(winesNorm, 6)$withindiff
wss[7] <- kmodes(winesNorm, 7)$withindiff
wss[8] <- kmodes(winesNorm, 8)$withindiff
wss[9] <- kmodes(winesNorm, 9)$withindiff
wss[10] <- kmodes(winesNorm, 10)$withindiff

# Total within-cluster sum of squares vs Choice of k
p4 <- qplot(1:10, wss, geom=c("point", "line"),
            xlab="Number of clusters", ylab="withindiff") +
  scale_x_continuous(breaks=seq(0, 10, 1)) +
  theme_bw()

# Subplot
grid.arrange( p4, ncol=1)
#####################################################################
#######################################k-Mode clustering##############################
# Execution of k-mode with k=3
set.seed(1234)

wines_kmode3 <- kmodes(winesNorm, 3, iter.max = 50, weighted = FALSE)

# Cluster to which each point is allocated
wines_kmode3 $cluster

# Cluster modes
wines_kmode3$modes

# Cluster size
wines_kmode3 $size

# Between-cluster sum of squares
wines_kmode3 $betweenss

# Within-cluster sum of squares
wines_kmode3 $withindiff

# Within-cluster sum of squares
wines_kmode3 $tot.withindiff

# Iterations
wines_kmode3 $iterations

# Mean values of each cluster
aggregate(wines, by=list(wines_kmode3$cluster), mode)

# Clustering 
ggpairs(cbind(wines, Cluster=as.factor(wines_kmode3$cluster)),
        columns=1:6, aes(colour=Cluster, alpha=0.5),
        lower=list(continuous="points"),
        upper=list(continuous="blank"),
        axisLabels="none", switch="both") +
  theme_bw()


#checking
table(data_$Class,wines_kmode3$cluster)