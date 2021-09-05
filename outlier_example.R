library(ggplot2)
library(tidyverse)

set.seed(1)

# generate 100 observations from the standard normal
x<-rnorm(100)

# add 100 to one single observation (an error or outlier) 
x_error=x
max_index=which(x_error==max(x_error))
x_error[max_index]=max(x_error)+10

# --------- plot histogram
# par(mfrow=c(1,2)) # plot two figures in one
# 
# #set breaks for the histogram
# x_breaks=seq(floor(min(x)), ceiling(max(x)), by=0.5)
# x_error_breaks=seq(floor(min(x_error)), ceiling(max(x_error)), by=0.5)
# 
# # set x axis limits
# my_lim=c(floor(min(x)), ceiling(max(x_error)))
#           
# h_x= hist(x,
#      main = "Original Data",
#      xlim=my_lim,
#      breaks=x_breaks,
#      xlab=""
#      )
# 
# hist(x_error,
#      main= "Corrupted Data",
#      xlim=my_lim, 
#      breaks=x_error_breaks,
#      xlab=""
#      )

# ------------- plot box plot

par(mfrow=c(1,1))

#boxplot(x, x_error,names=c("Original","Corrupted"))

types=c(rep("Original",100),rep("Corrupted",100))
values=c(x,x_error)
df= data.frame(types,values)
colnames(df)<- c("Type","Value")
df$Type <- factor(df$Type,
                       levels = c("Original","Corrupted"),ordered = TRUE)

ggplot(df, aes(x = Type, y = Value, fill=Type)) +
  geom_boxplot(outlier.colour = "red",
               outlier.size = 3)


ggplot(df, aes(x = Type, y = Value, fill=Type)) +
  geom_bar(stat = "summary", fun = "mean") +
  ylab("Mean")




summary(x)
summary(x_error)
