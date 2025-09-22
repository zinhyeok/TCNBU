#' Change-Point search by graph-based WBS
#' @description This function find all potential change-points in a sequence by graph-based scan statistics and wild binary segmentation.
#' @param y A n x d matrix to be scanned with n observations and d dimensions.
#' @param s Position where the scan starts.
#' @param e Position where the scan ends.
#' @param alpha Significance level for each scan statistic.
#' @param L Number of generated intervals in each recursion.
#' @param minLen Minimum length of generated intervals.
#' @param stat The scan statistic to be computed.
#'
#'      "g" specifies the generalized edge-count scan statistic;
#'
#'      "o" specifies the original edge-count scan statistic;
#'
#'      "w" specifies the weighted edge-count scan statistic;
#'
#'      "m" specifies the max-type edge-count scan statistic.
#' @param graph The type of similarity graphs.
#'
#'     "mst" specifies the minimum spanning tree;
#'
#'     "knn" specifies the nearest neighbor graph.
#' @param kMax Max k of the similarity graph (k-mst or knn).
#' @param cutoff Cutoff percentage at the beginning and end of each scan.
#' @param distType The distance measure to be used in the \code{\link[stats]{dist}} function.
#' @param maxLen Maximum length of generated intervals.
#' @param ... Arguments to be passed to \code{\link[gSeg]{gseg1}}.
#' @return Scanned candidate change-points.
#' @examples
#' set.seed(1)
#' y <- rbind(
#'   matrix(rnorm(200), ncol = 10), matrix(rexp(200), ncol = 10),
#'   matrix(rnorm(200), ncol = 10), matrix(rexp(200), ncol = 10)
#' )
#' gWBS(y, s = 1, e = 80, alpha = 0.01, L = 50, graph = "mst")
#' @import stats
#' @import utils


#' @export
gWBS <- function(y, s = 1, e = nrow(y), alpha = 0.01, L = 100, minLen = 10, stat = "g", graph = "mst", kMax = 30, cutoff = 0.05, distType = "euclidean", maxLen = 1000, ...) {
  .g.WBS(
    y = y,
    s = s,
    e = e,
    alpha = alpha,
    L = L,
    minLen = minLen,
    stat = stat,
    cutoff = cutoff,
    graph = graph,
    kMax = kMax,
    distType = distType,
    maxLen = maxLen, ...
  )
}

.g.WBS <- function(y,
                   s = 1,
                   e = nrow(y),
                   alpha = 0.01,
                   L = 100,
                   minLen = 10,
                   stat = "g",
                   cutoff = 0.05,
                   graph = "mst",
                   kMax = 30,
                   distType = "euclidean",
                   maxLen = 1000,
                   mainFunc = TRUE,
                   temp.env = numeric(0)) {
  len <- e - s + 1
  if (mainFunc == TRUE) {
    temp.env <- new.env(parent = emptyenv())
    temp.env$tauhat <- numeric(0)
    if (len^2 * ncol(y) > 1e11) {
      message(
        "The dataset is very large, and it might take hours to run this. Are you sure to continue (y/n)?"
      )
      inp <- readline()
      if (!inp %in% c("y", "Y", "yes", "YES", "Yes")) {
        return("The function has stopped.")
      }
    }
  }
  if (stat == "o") {
    argchar <- "ori"
  } else if (stat == "w") {
    argchar <- "weighted"
  } else if (stat == "m") {
    argchar <- "max.type"
  } else {
    argchar <- "generalized"
  }
  if (len < minLen) {
    return(NULL)
  } else {
    if (L >= (len - minLen + 1) * (len - minLen + 2) / 2) {
      FTM <- matrix(0, ncol = 2, nrow = 0)
      i <- len - minLen + 1
      for (l in s:(e - minLen + 1)) {
        FTM <- rbind(FTM, matrix(c(rep(l, i), (l + minLen - 1):e), ncol = 2, nrow = i))
        i <- i - 1
      }
    } else {
      FTM <- matrix(0, ncol = 2, nrow = L)
      prob1 <- pmin(e - (s:(e - minLen + 1)), maxLen - 1) - (minLen - 2)
      prob1 <- prob1 / sum(prob1)
      FTM[, 1] <- sample(s:(e - minLen + 1), L, prob = prob1, replace = TRUE)
      FTM[, 2] <- sapply(FTM[, 1], function(x) {
        rn <- (x + minLen - 1):min(e, x + maxLen - 1)
        rn[sample.int(length(rn), 1)]
      })
      if (len <= maxLen) {
        FTM <- rbind(FTM, c(s, e))
      }
    }
  }
  all.pvalue <- numeric(dim(FTM)[1])
  all.position <- numeric(dim(FTM)[1])
  for (l in 1:dim(FTM)[1]) {
    m <- FTM[l, 2] - FTM[l, 1] + 1
    # print(FTM[l, ])
    if (graph == "nng") {
      E <- igraph::get.edgelist(igraph::as.undirected(cccd::nng(y[FTM[l, 1]:FTM[l, 2], ], k = min(
        floor(sqrt(m)), kMax
      ), method = distType), mode = "collapse"))
    } else {
      E <- ade4::mstree(dist(y[FTM[l, 1]:FTM[l, 2], ], method = distType), ngmax = min(floor(sqrt(m)), kMax))
    }
    Etable <- table(E)
    flag <- 0
    if (max(Etable) == (m - 1)) {
      flag <- 1
    } else if ((max(Etable) - min(Etable)) == 0) {
      flag <- 1
    } else {
      r <- try(gSeg::gseg1(
        m,
        E,
        statistics = stat,
        n0 = cutoff * m,
        n1 = (1 - cutoff) * m
      ))
      if (inherits(r, "try-error")) {
        flag <- 1
      }
    }
    if (flag == 1) {
      all.pvalue[l] <- 1
      all.position[l] <- 1
    } else {
      all.pvalue[l] <- eval(parse(text = paste("r$pval.appr$", argchar, sep = "")))
      all.position[l] <- eval(parse(text = paste(
        "r$scanZ$", argchar, "$tauhat",
        sep = ""
      )))
    }
  }
  bestindex <- which.min(all.pvalue)
  if (min(all.pvalue) < alpha) {
    newtau <- all.position[bestindex]
    cat(c(s, e, newtau))
    temp.env$tauhat <- c(temp.env$tauhat, newtau + FTM[bestindex, 1] - 1)
    .g.WBS(
      y = y,
      s = s,
      e = newtau + FTM[bestindex, 1] - 1,
      alpha = alpha,
      L = L,
      minLen = minLen,
      stat = stat,
      cutoff = cutoff,
      graph = graph,
      kMax = kMax,
      distType = distType,
      maxLen = maxLen,
      mainFunc = FALSE,
      temp.env = temp.env
    )
    .g.WBS(
      y = y,
      s = newtau + FTM[bestindex, 1],
      e = e,
      alpha = alpha,
      L = L,
      minLen = minLen,
      stat = stat,
      cutoff = cutoff,
      graph = graph,
      kMax = kMax,
      distType = distType,
      maxLen = maxLen,
      mainFunc = FALSE,
      temp.env = temp.env
    )
  }
  if (mainFunc == TRUE) {
    tauhat <- temp.env$tauhat
    rm(temp.env)
    return(tauhat)
  } else {
    return(NULL)
  }
}
