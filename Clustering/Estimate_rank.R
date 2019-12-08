estimateRank <- function(data, nb_permutation = 50, interval = seq(2,9), method='snmf/r', nrun=30, beta= 0.01, true_values, plot= "plo", type_plot = "IC") {

  sils= matrix(0,nb_permutation,length(interval))
  cc= matrix(0,nb_permutation,length(interval))
  len=length(interval)
  for (i in 1:len)
  {
    for (j in 1:nb_permutation)
    {  
    nf = randomize(data)
    gr= nmf(t(nf),interval[i], method='snmf/r', nrun=nrun, beta= beta)
    #print(silhouette(gr, "consensus"))
    cc[j,i] = cophcor(gr)
    }
  }
  
  
  if (plot == "plot")
  {
    if (type_plot == "IC")
    {
    inf = rep(0,len)
    sup = rep(0,len)
    val = rep(0,len)
    m = rep(0,len)
    
    for (i in 1:len)
    {
      mm=mean(cc[,i])
      dd=sd(cc[,i])
      error = 1.96*dd/sqrt(nb_permutation)
      m[i] = mm
      inf[i] = mm-error
      sup[i] = mm+error
      val[i] = true_values[i]
    }
    a= 0.5
    b=1.5    
    boxplot(cc,col='#F0EEE5', names=c(interval), xlab='Number of clusters',ylab="Cophenetic Correlation")

    for (i in 1:len)
    {
      lines(c(a,b),c(inf[i],inf[i]),col=4)
      lines(c(a,b),c(m[i],m[i]),col=3,lwd=2)
      lines(c(a,b),c(sup[i],sup[i]),col=4)
      lines(c(a,b),c(val[i],val[i]),col=2,lwd=2)
      a= a+1
      b=b+1
    }
    legend("bottomleft", c("True_CC","95% CI", "Mean"), lty=1,col = c(2,4,3),bty ="n")
  }
    if (type_plot == 'jitter')
    {
      
      stripchart(cc,vertical = TRUE,  
                 method = "jitter", add = TRUE, pch = 20, col = 'blue')
    }
  
    }
  return(cc)
  }
  
