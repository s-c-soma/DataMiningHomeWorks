# Load libraries
library(tidyverse)
library(corrplot)
library(gridExtra)
library(GGally)
library(knitr)

# Read the stats
wines <- read.csv("C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/HW4_Submission/wine.csv")

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