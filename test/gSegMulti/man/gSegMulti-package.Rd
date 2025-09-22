\name{gSegMulti-package}
\alias{gSegMulti-package}
\title{Multiple change-point detection framework by graph-based method}
\description{This package uses edge-count scan statistic and greedy algorithms to estimate position of change-points in high dimensional data or non-Euclidean data. You can use \code{\link[gSegMulti]{gSegMulti}} to conduct the whole procedure conveniently.

The estimation involves two steps. First we use a greedy method to search for candidate change-point. If you prefer wild binary segmentation for an extensive search, the funtion \code{\link{gWBS}} should be use. If you prefer seeded binary segmentation for an fast search, the function \code{\link{gSBS}} can be used.

The next step involves using a goodness-of-fit measure expanded pseudo-BIC and backward elimination for pruning. You may use \code{\link{gBE}} for this model selection procedure.

Finally, we provide a visualization tool called change-point dendrogram to visualize model selection result form the second step. It can be accessed by \code{\link{cpdendrogram}}.
}
\author{
Yuxuan Zhang and Hao Chen

Maintainer: Hao Chen (hxchen@ucdavis.edu)
}
\references{
Zhang, Yuxuan, and Hao Chen. "Graph-based multiple change-point detection." arXiv preprint arXiv:2110.01170 (2021).
}

\seealso{
  \code{\link{gSegMulti}}, \code{\link{gWBS}}, \code{\link{gSBS}}, \code{\link{gBE}}, \code{\link{cpdendrogram}}
}
\examples{
library(gSegMulti)
n = 150
rho = 0.3
d = 100
Sigma = matrix(0, ncol = d, nrow = d)
for(i in 1:d){
  for(j in 1:d){
    Sigma[i,j] = rho^abs(i - j)
}}
y = matrix(0, ncol = d, nrow = 0)
y <- rbind(
   matrix(rnorm(200), ncol = 10), matrix(rexp(200), ncol = 10),
   matrix(rnorm(200), ncol = 10), matrix(rexp(200), ncol = 10)
)

# step 1: searching candidate change-points by WBS

step1 = gWBS(y)

# step 2: pruning change-points by backward elimination (model selection)

step2 = gBE(y, step1, detail = TRUE)

print(step2)
cpdendrogram(y, step2) # plot change-point dendrogram
}



