#' Change-Point search by graph-based SBS
#' @description This function find all potential change-points in a sequence by graph-based scan statistics and seeded binary segmentation.
#' @param y A n x d matrix to be scanned with n observations and d dimensions.
#' @param s Position where the scan starts.
#' @param e Position where the scan ends.
#' @param alpha Significance level for each scan statistic.
#' @param decay.a Decay parameter used in SBS (between 0.5 and 1).
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
#' gSBS(y, s = 1, e = 80, alpha = 0.01, decay.a = sqrt(0.5), graph = "mst")
#' @import stats
#' @import utils
#' @export
gSBS <-
  function(y,
           s = 1,
           e = nrow(y),
           alpha = 0.01,
           decay.a = sqrt(0.5),
           minLen = 10,
           stat = "g",
           graph = "mst",
           kMax = 30,
           cutoff = 0.05,
           distType = "euclidean",
           maxLen = 1000,
           ...) {
    .g.SBS(
      y = y,
      s = s,
      e = e,
      alpha = alpha,
      decay.a = decay.a,
      minLen = minLen,
      stat = stat,
      cutoff = cutoff,
      graph = graph,
      kMax = kMax,
      distType = distType,
      maxLen = maxLen,
      ...
    )
  }

.g.SBS <- function(y,
                   s = 1,
                   e = nrow(y),
                   alpha = 0.01,
                   decay.a = sqrt(0.5),
                   minLen = 10,
                   stat = "g",
                   cutoff = 0.05,
                   graph = "mst",
                   kMax = 30,
                   distType = "euclidean",
                   maxLen = 1000,
                   mainFunc = TRUE,
                   temp.env = numeric(0)) {
  if (stat == "o") {
    argchar <- "ori"
  } else if (stat == "w") {
    argchar <- "weighted"
  } else if (stat == "m") {
    argchar <- "max.type"
  } else {
    argchar <- "generalized"
  }
  n <- e - s + 1
  if (n < minLen) {
    return(NULL)
  }
  if (mainFunc == TRUE) {
    temp.env <- new.env(parent = emptyenv())
    temp.env$tauhat <- numeric(0)
    if (n^2 * ncol(y) > 1e12) {
      message(
        "The dataset is very large, and it might take hours to run this. Are you sure to continue (y/n)?"
      )
      inp <- readline()
      if (!inp %in% c("y", "Y", "yes", "YES", "Yes")) {
        return("The function has stopped.")
      }
    }
    temp.env$FTM <- matrix(c(s, e), ncol = 2, byrow = TRUE)
    temp.env$all.pvalue <- numeric(dim(temp.env$FTM)[1])
    temp.env$all.position <- numeric(dim(temp.env$FTM)[1])
    for (k in 2:ceiling(log(e, base = 1 / decay.a))) {
      lk <- e * decay.a^(k - 1)
      if (lk >= (minLen - 1) & lk < (maxLen - 1)) {
        nk <- 2 * ceiling((1 / decay.a)^(k - 1)) - 1
        sk <- (e - lk) / (nk - 1)
        Ik <- matrix(c(floor((0:(
          nk - 1
        )) * sk), ceiling((0:(
          nk - 1
        )) * sk + lk)),
        ncol = 2,
        byrow = FALSE
        )
        temp.env$FTM <- rbind(temp.env$FTM, Ik)
      }
    }
    temp.env$FTM[, 1] <- ifelse(temp.env$FTM[, 1] >= 1, temp.env$FTM[, 1], 1)
    temp.env$FTM[, 2] <- ifelse(temp.env$FTM[, 2] <= e, temp.env$FTM[, 2], e)
    for (l in 1:dim(temp.env$FTM)[1]) {
      # print(temp.env$FTM[l,])
      m <- temp.env$FTM[l, 2] - temp.env$FTM[l, 1] + 1
      if (graph == "nng") {
        E <- igraph::get.edgelist(igraph::as.undirected(cccd::nng(
          y[temp.env$FTM[l, 1]:temp.env$FTM[l, 2], ],
          k = min(floor(sqrt(m)), kMax), method = distType
        ), mode = "collapse"))
      } else {
        E <- ade4::mstree(dist(y[temp.env$FTM[l, 1]:temp.env$FTM[l, 2], ], method = distType), ngmax = min(floor(sqrt(m)), kMax))
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
        temp.env$all.pvalue[l] <- 1
        temp.env$all.position[l] <- 1
      } else {
        temp.env$all.pvalue[l] <- eval(parse(text = paste("r$pval.appr$", argchar, sep = "")))
        temp.env$all.position[l] <- eval(parse(text = paste(
          "r$scanZ$", argchar, "$tauhat",
          sep = ""
        )))
      }
    }
  }

  Mse <- which((temp.env$FTM[, 1] >= s) & temp.env$FTM[, 2] <= e)
  pvalue <- temp.env$all.pvalue[Mse]
  bestindex <- which.min(pvalue)

  if(graph=='mst'){
    E<-ade4::mstree(dist(y[s:e,], method = distType), ngmax = min(floor(sqrt(n)), kMax))
    r<-gSeg::gseg1(n,E,
                   statistics = stat,
                   n0 = cutoff * n,
                   n1 = (1 - cutoff) * n)
  }else if(graph=='nng'){
    E <- igraph::get.edgelist(igraph::as.undirected(cccd::nng(
      y[s:e, ],
      k = min(floor(sqrt(n)), kMax), method = distType
    ), mode = "collapse"))
    r<-gSeg::gseg1(n,E,statistics = stat,n0=cutoff*n,n1=(1-cutoff)*n)
  }

  if(min(pvalue)>eval(parse(text=paste('r$pval.appr$',argchar,sep='')))){
    if(eval(parse(text=paste('r$pval.appr$',argchar,sep='')))<alpha){
      newtau <-eval(parse(text=paste('r$scanZ$',argchar,'$tauhat',sep='')))
      temp.env$tauhat <- c(temp.env$tauhat, newtau + s - 1)
      .g.SBS(
        y = y,
        s = s,
        e = newtau + s - 1,
        alpha = alpha,
        stat = stat,
        cutoff = cutoff,
        kMax = kMax,
        graph = graph,
        minLen = minLen,
        decay.a = decay.a,
        maxLen = maxLen,
        mainFunc = FALSE,
        temp.env = temp.env,
        distType = distType
      )
      .g.SBS(
        y = y,
        s = newtau + s,
        e = e,
        alpha = alpha,
        stat = stat,
        cutoff = cutoff,
        kMax = kMax,
        graph = graph,
        minLen = minLen,
        decay.a = decay.a,
        maxLen = maxLen,
        mainFunc = FALSE,
        temp.env = temp.env,
        distType = distType
      )
    }
  }else{
    if (min(pvalue) < alpha) {
      newtau <- temp.env$all.position[Mse[bestindex]]
      temp.env$tauhat <- c(temp.env$tauhat, newtau + temp.env$FTM[Mse[bestindex], 1] - 1)
      .g.SBS(
        y = y,
        s = s,
        e = newtau + temp.env$FTM[Mse[bestindex], 1] - 1,
        alpha = alpha,
        stat = stat,
        cutoff = cutoff,
        kMax = kMax,
        graph = graph,
        minLen = minLen,
        decay.a = decay.a,
        maxLen = maxLen,
        mainFunc = FALSE,
        temp.env = temp.env,
        distType = distType
      )
      .g.SBS(
        y = y,
        s = newtau + temp.env$FTM[Mse[bestindex], 1],
        e = e,
        alpha = alpha,
        stat = stat,
        cutoff = cutoff,
        kMax = kMax,
        graph = graph,
        minLen = minLen,
        decay.a = decay.a,
        maxLen = maxLen,
        mainFunc = FALSE,
        temp.env = temp.env,
        distType = distType
      )
    }
  }
  if (mainFunc == TRUE) {
    tauhat <- temp.env$tauhat
    rm(temp.env)
    return(tauhat)
  } else {
    return(NULL)
  }
}
