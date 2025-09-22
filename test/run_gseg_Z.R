# run_gseg_Z.R
# if (!require("ade4")) install.packages("ade4", dependencies = TRUE)
library(ade4)
# if (!require("gSeg")) install.packages("gSeg", dependencies = TRUE)
library(gSeg)
# if (!require("jsonlite")) install.packages("jsonlite", dependencies = TRUE)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_gseg_Z.R input.csv output.json")
}
input_file <- args[1]
output_file <- args[2]
alpha <- as.numeric(args[3]) 
k <- as.numeric(args[4])  # k value for the graph, if provided 

# load data and construct MST graph
data <- as.matrix(read.csv(input_file, header = FALSE))
n <- nrow(data)
dists <- 	dist(data, method = "euclidean")
E <- mstree(dists, k)

# run gseg1 (generalized statistic), p-value not needed
res <- gseg1(n, E,
             statistics = "g",
             pval.appr = TRUE,
             skew.corr = TRUE)

tauhat = res$scanZ$generalized$tauhat
P_val = res$pval.appr
Slst = res$scanZ$generalized$S

# if (P_val > alpha) {
#   tauhat <- NULL
# }

# extract change-point estimate and statistic
output <- list(
  tauhat = tauhat,
  pval = P_val,
  Slst = Slst
)
# output <- res

write(toJSON(output, auto_unbox = TRUE, digits = 10), file = output_file)
system.file(package = "ade4")