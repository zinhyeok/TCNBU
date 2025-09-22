#' Generate Change-point Dendrogram
#' @description Plotting a change-point dendrogram according to the result of graph-based backward elimination.
#' @param y A n x d matrix to be scanned with n observations and d dimensions.
#' @param BEresult Detailed result list returned by \code{\link{gBE}}.
#' @param startMax If the change-point dendrogram starting from the position where goodness-of-fit statistic is maximized.
#' @param flat If the height of nodes are set to be at least the height of their children.
#' @param shortLabel If the dendrogram only shows starting points of intervals.
#' @param title Title of the change-point dendrogram.
#'
#' @examples
#' set.seed(1)
#' y <- rbind(
#'   matrix(rnorm(200), ncol = 10), matrix(rexp(200), ncol = 10),
#'   matrix(rnorm(200), ncol = 10), matrix(rexp(200), ncol = 10)
#' )
#' cpdendrogram(y, gBE(y, c(20, 30, 40, 60), c = 2, kMax = 2, detail = TRUE))
#'
#' @export
cpdendrogram <- function(y, BEresult, startMax = TRUE, flat = TRUE, shortLabel = FALSE, title = "change-point dendrogram") {
  argchar <- "generalized"
  argchar2 <- "generalized"
  tauseq <- c(BEresult[[3]], BEresult[[1]][1])
  gofstat <- BEresult[[2]]
  if (startMax == TRUE & which.max(gofstat) != 1) {
    tauseq <- tauseq[-(1:(which.max(gofstat) - 1))]
    gofstat <- gofstat[-(1:(which.max(gofstat) - 1))]
  }
  n <- dim(y)[1]
  tauseq.short <- tauseq
  sort.tauseq.short <- sort(tauseq.short)
  m <- length(tauseq.short)
  tauseq <- c(0, sort(tauseq), n)
  denlist <- list()
  if (shortLabel == TRUE) {
    denlist$labels <- head(tauseq, m + 1) + 1
  } else {
    denlist$labels <- paste(head(tauseq, m + 1) + 1, "-", tail(tauseq, m + 1), sep = "")
  }
  denlist$order <- 1:(m + 1)
  denlist$height <- numeric(0)
  cur.level <- matrix(c(1:(m + 1), rep(NA, (m + 1))), nrow = 2, byrow = T)
  tau.order <- order(tauseq.short)
  tau.rank <- rank(tauseq.short)
  denlist$merge <- matrix(numeric(0), ncol = 2, nrow = 0)
  for (i in 1:m) {
    cur.merge.posi <- tau.rank[i]
    if (is.na(cur.level[2, cur.merge.posi]) & is.na(cur.level[2, cur.merge.posi + 1])) {
      denlist$merge <- rbind(denlist$merge, c(-cur.merge.posi, -(cur.merge.posi + 1)))
      cur.level[2, c(cur.merge.posi, cur.merge.posi + 1)] <- i
    } else if ((!is.na(cur.level[2, cur.merge.posi])) & (!is.na(cur.level[2, cur.merge.posi + 1]))) {
      denlist$merge <- rbind(denlist$merge, cur.level[2, c(cur.merge.posi, cur.merge.posi + 1)])
      cur.level[2, cur.level[2, ] %in% c(cur.level[2, c(cur.merge.posi, cur.merge.posi + 1)])] <- i
    } else if (is.na(cur.level[2, cur.merge.posi])) {
      denlist$merge <- rbind(denlist$merge, c(-cur.merge.posi, cur.level[2, (cur.merge.posi + 1)]))
      cur.level[2, cur.level[2, ] == cur.level[2, (cur.merge.posi + 1)]] <- i
      cur.level[2, cur.merge.posi] <- i
    } else {
      denlist$merge <- rbind(denlist$merge, c(-cur.merge.posi - 1, cur.level[2, cur.merge.posi]))
      cur.level[2, cur.level[2, ] == cur.level[2, cur.merge.posi]] <- i
      cur.level[2, cur.merge.posi + 1] <- i
    }
    inte.index <- (tauseq[min(which(cur.level[2, ] == i))] + 1):tauseq[max(which(cur.level[2, ] == i)) + 1]
    o <- length(inte.index)
  }
  denlist$height <- -gofstat
  if (flat == TRUE) {
    for (i in 1:dim(denlist$merge)[1]) {
      if (sum(denlist$merge[i, ] > 0) == 1) {
        nodeind <- denlist$merge[i, ][denlist$merge[i, ] > 0]
        if (denlist$height[i] < denlist$height[nodeind]) {
          denlist$height[i] <- denlist$height[nodeind]
        }
      } else if (sum(denlist$merge[i, ] > 0) == 2) {
        heightold <- max(denlist$height[denlist$merge[i, ]])
        if (denlist$height[i] < heightold) {
          denlist$height[i] <- heightold
        }
      }
    }
  }
  class(denlist) <- "hclust"
  plot(denlist, xlab = "subsequence", ylab = "- ep-BIC", main = title)
}
