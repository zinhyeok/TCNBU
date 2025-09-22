
# gSegMulti

<!-- badges: start -->
<!-- badges: end -->

This package uses graph-based scan statistics to detect multiple
change-points in high-dimensional or non-Euclidean data. It involves the
use of edge-count scan statistics, wild binary segmentation or seeded
binary segmentation and model selection algorithms.

The method is composed of two steps. First, use wild binary segmentation
or seeded binary segmentation to scan for a pool of candidate
change-points. Next, use backward elimination to prune candidate
change-points. The second step could be visualized by a change-point
dendrogram.

## Example

``` r
library(gSegMulti)
set.seed(1)
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
```

``` r
print(step2)
#> $tauhat
#> [1] 60 41 21
#> 
#> $gofSeq
#> [1] 184.3264 276.2822 126.9812  21.4385
#> 
#> $mergeSeq
#> [1] 29 21 41
cpdendrogram(y, step2) # plot change-point dendrogram
```

<img src="man/figures/README-unnamed-chunk-2-1.png" width="100%" />

## Reference

Zhang, Yuxuan, and Hao Chen. “Graph-based multiple change-point
detection.” arXiv preprint arXiv:2110.01170 (2021).
