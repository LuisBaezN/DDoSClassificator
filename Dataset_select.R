#/////////////////////////////////////Data set prueba///////////////////////////////////////

#01-12: DrDoS_DNS DrDoS_NetBIOS Syn TFTP UDPLag
#03-11: LDAP MSSQL NetBIOS Portmap Syn UDP UDPLag

num_fil <- 20000
data <- read.csv("L:/etern/Blue disk/PC2/Criptografia/Proyecto/01-12/DrDoS_DNS.csv", header=FALSE, nrows = num_fil, skip = 2) #, skip = 2

summary(data[,88])

#////////////////////////////////////////Exploracion////////////////////////////////////////

#x <- 1:nrow(data)
#x <- 1:100

#////////////////////////////////////////Seleccion//////////////////////////////////////////

atrib <- c(3,4,5,6,7,9,11,13,21,22,23,24,42,62,80,82,88)
#SourceIP,SourcePort,DestinationIP,DestinationPort,Protocol
#FlowDuration,TotalBackwardPackets,TotalLenghtofBwdPackets,BwdPacketLengthStd,Flowbytes/s,FlowPackets/s
#//////FlowIATMean,FwdHeaderLength,AvgBwdSegmentsize,Active Min,idle Mean,Label

data_s <- data[,atrib]

#/////////////////////////////////////Data set UDPLag///////////////////////////////////////
data_s <- na.omit(data_s)
num_col <- ncol(data_s)
data <- matrix(nrow = num_fil, ncol = num_col)

for (i in 1:nrow(data_s))
{
  if (data_s[i,10] == "Inf" || data_s[i,11] == "Inf")
    i
  else
  {
    for (j in 1:num_col)
      data[i,j] <- data_s[i,j]
  }
}	

data <- na.omit(data)
rm(data_s)
rm(atrib)
#solo flujo de bits y paquetes por segundo son continuos hasta 70,000

#/////////////////////////////////Construccion///////////////////////////////////////
data_f <- matrix( ncol = ncol(data))

c2 <- 0
c3 <- 0

for (i in 1:nrow(data)) 
{
  if (data[i, 17] == 1)
  {
    data_f <- rbind(data_f,data[i,])
  }
  else
  {
    if (data[i, 17] == 2)
    {
      c2 <- c2 + 1
      if (c2 == 14)         #/////////salto/////////
      {
        data_f <- rbind(data_f,data[i,])
        c2 <- 0
      }
    }
      
    else
    {
      if (data[i, 17] == 3)
      {
        c3 <- c3 + 1
        if (c3 == 14)      #/////////saltos/////////
        {
          data_f <- rbind(data_f,data[i,])
          c3 <- 0
        }
      }
    }
  }
}

data_f <- na.omit(data_f)

for (i in 1:nrow(data_f)) 
{
  if (data_f[i,17] == 2)
    data_f[i,17] <- 4      #//////////////////Ataques//////////////////////
  else
  {
    if (data_f[i,17] == 3)
      data_f[i,17] <- 4     #//////////////////Ataques//////////////////////
  }
}

x <- 1:nrow(data_f)
plot(x,data_f[,17], type = 'l')


write.table(data_f, "Database_f_v2.csv", sep = ",", col.names = F, append = T,row.names = F)








#////////////////////////////////////////otro dataset///////////////////////////////////
datos <- c(4, 6, 7, 9, 10, 11, 23, 27) 
time <- data[,8] 
label <- data[,88]
label_n <- data.matrix(label)

for (i in 1:nrow(data))
{
  if (label_n[i] == "BENIGN")
  {
    label_n[i] <- 0
  }
  else-if (label_n[i] == "UDP-lag")
  {
    label_n[i] <- 1
  }
  else-if (label_n[i] == "WebDDoS")
  {
    label_n[i] <- 2
  }
  else
  {
    i
  }
}

data_n <- data[,datos]


data_u <- cbind(time, data_n, label_n)#, label_n

write.table(data_u, "UDPLag_red.csv", sep = ",", col.names = F, append = T,row.names = F)
#write.csv(data_u,"UDPLag_red.csv", col.names = FALSE, row.names = FALSE)
