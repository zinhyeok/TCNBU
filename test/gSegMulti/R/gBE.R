#' Change-Point selection by graph-based backward elimination
#' @description This function prunes candidate change-points and choose the model with the largest expanded pseudo-BIC values.
#' @param y A n x d matrix to be scanned with n observations and d dimensions.
#' @param tauhat A vector of candidate change-points to be selected.
#' @param c Penalty multiplier.
#' @param graph The type of similarity graphs.
#'
#'     "mst" specifies the minimum spanning tree;
#'
#'     "knn" specifies the nearest neighbor graph.
#' @param kMax Max k of the similarity graph (k-mst or knn).
#' @param distType The distance measure to be used in the \code{\link[stats]{dist}} function.
#' @param minCpNum Minimum number of change-points.
#' @param maxSeq Maximum length of sequence to be tested.
#' @param detail Return detailed information.
#'
#' @return If \code{detail = FALSE}, the function returns selected change-points. If \code{detail = TRUE}, the function returns a list.
#' \item{tauhat}{Selected change-points.}
#' \item{gofSeq}{Goodness-of-fit statistic values in each step of backward elimination.}
#' \item{mergeSeq}{Order of candidate change-points removed in backward elimination.}
#'
#' @examples
#' set.seed(1)
#' y <- rbind(
#'   matrix(rnorm(200), ncol = 10), matrix(rexp(200), ncol = 10),
#'   matrix(rnorm(200), ncol = 10), matrix(rexp(200), ncol = 10)
#' )
#' gBE(y, c(20, 30, 40, 60), c = 2, kMax = 2, detail = TRUE)
#' @import stats
#' @import utils
#' @export
gBE <- function(y,
                tauhat,
                c = 2,
                graph = "mst",
                kMax = 5,
                distType = "euclidean",
                minCpNum = 1,
                maxSeq = 5000,
                detail = FALSE) {
  sim.g <- function(y, s, e) {
    if (graph == "nng") {
      return(igraph::get.edgelist(igraph::as.undirected(
        cccd::nng(y[s:e, ], k = min(floor(sqrt(
          e - s + 1
        )), kMax), method = distType),
        mode = "collapse"
      )))
    } else {
      return(ade4::mstree(dist(y[s:e, ], method = distType), min(floor(sqrt(
        e - s + 1
      )), kMax)))
    }
  }
  m <- length(tauhat)
  if (m <= minCpNum) {
    return(tauhat)
  }
  argchar <- "generalized"
  tauhat <- sort(tauhat)
  n <- dim(y)[1]

  mergetau <- numeric(0)
  gofstat <- numeric(0)
  gofoverall <- numeric(0)
  tauhat <- c(0, tauhat, n)
  count <- 0
  for (j in 1:m) {
    flag <- 0
    i <- j - count
    EE <- sim.g(y, tauhat[i] + 1, tauhat[i + 2])
    EEtable <- table(EE)
    if (max(EEtable) == (tauhat[i + 2] - tauhat[i] - 1)) {
      flag <- 1
    } else if (max(EEtable) == min(EEtable)) {
      flag <- 1
    }
    if (flag == 1) {
      # print('flag')
      # cat(c(j, "\n"))
      tauhat <- tauhat[-(i + 1)]
      count <- count + 1
      flag <- 0
      if (i != 1) {
        EEl <- sim.g(y, tauhat[i - 1] + 1, tauhat[i + 1])
        rl <- gTests::g.tests(
          EE,
          1:(tauhat[i] - tauhat[i - 1]),
          (tauhat[i] - tauhat[i - 1] + 1):(tauhat[i + 1] - tauhat[i - 1]),
          "g"
        )
        gofstat[i - 1] <- eval(parse(text = paste(
          "rl$", argchar, "$test.statistic",
          sep = ""
        )))
      }
    } else {
      r <- gTests::g.tests(
        EE,
        1:(tauhat[i + 1] - tauhat[i]),
        (tauhat[i + 1] - tauhat[i] + 1):(tauhat[i + 2] - tauhat[i]),
        "g"
      )
      gofstat[i] <- eval(parse(text = paste(
        "r$", argchar, "$test.statistic",
        sep = ""
      )))
    }
  }
  gofoverall[1] <- sum(gofstat) - c * (length(tauhat) - 2) * log(n) - sum(log(diff(tauhat)))

  gofstat_back <- gofstat

  if (length(tauhat) >= (minCpNum + 2)) {
    change <- numeric(0)
    getchange <- function(tauhat, gofstat, j) {
      if (j == 1) {
        EE <- sim.g(y, 1, tauhat[4])
        r <- gTests::g.tests(EE, 1:tauhat[3], (tauhat[3] + 1):tauhat[4], "g")
        return(eval(parse(
          text = paste("r$", argchar, "$test.statistic", sep = "")
        )) - sum(gofstat[1:2]))
      } else if (j == (length(tauhat) - 2)) {
        EE <- sim.g(y, tauhat[j - 1] + 1, n)
        r <- gTests::g.tests(
          EE,
          1:(tauhat[j] - tauhat[j - 1]),
          (tauhat[j] - tauhat[j - 1] + 1):(n - tauhat[j - 1])
        )
        return(eval(parse(
          text = paste("r$", argchar, "$test.statistic", sep = "")
        )) - sum(gofstat[(j - 1):j]))
      } else {
        EEl <- sim.g(y, tauhat[j - 1] + 1, tauhat[j + 2])
        rl <- gTests::g.tests(
          EEl,
          1:(tauhat[j] - tauhat[j - 1]),
          (tauhat[j] - tauhat[j - 1] + 1):(tauhat[j + 2] - tauhat[j - 1])
        )
        EEr <- sim.g(y, tauhat[j] + 1, tauhat[j + 3])
        rr <- gTests::g.tests(
          EEr,
          1:(tauhat[j + 2] - tauhat[j]),
          (tauhat[j + 2] - tauhat[j] + 1):(tauhat[j + 3] - tauhat[j])
        )
        return(eval(parse(
          text = paste("rl$", argchar, "$test.statistic", sep = "")
        )) + eval(parse(
          text = paste("rr$", argchar, "$test.statistic", sep = "")
        )) - sum(gofstat[(j - 1):(j + 1)]))
      }
    }
    for (j in 1:(length(tauhat) - 2)) {
      change[j] <- getchange(tauhat, gofstat, j)
    }
  } else {
    return(tauhat)
  }

  change_back <- change

  i <- 1
  while (length(tauhat) >= (minCpNum + 4) &&
    max(tauhat[4:length(tauhat)] - tauhat[1:(length(tauhat) - 3)]) < maxSeq) {
    maxind <- which.max(change)
    mergetau[i] <- tauhat[maxind + 1]
    gofstat <- gofstat[-maxind]
    tauhat <- tauhat[-(maxind + 1)]
    change <- change[-maxind]
    if (maxind == 1) {
      EE <- sim.g(y, 1, tauhat[3])
      r <- gTests::g.tests(EE, 1:tauhat[2], (tauhat[2] + 1):tauhat[3])
      gofstat[1] <- eval(parse(text = paste(
        "r$", argchar, "$test.statistic",
        sep = ""
      )))
      for (id in 1:2) {
        change[id] <- getchange(tauhat, gofstat, id)
      }
    } else if (maxind == (length(change) + 1)) {
      EE <- sim.g(y, tauhat[maxind - 1] + 1, n)
      r <- gTests::g.tests(
        EE,
        1:(tauhat[maxind] - tauhat[maxind - 1]),
        (tauhat[maxind] - tauhat[maxind - 1] + 1):(n - tauhat[maxind - 1])
      )
      gofstat[maxind - 1] <- eval(parse(text = paste(
        "r$", argchar, "$test.statistic",
        sep = ""
      )))
      for (id in c(maxind - 2, maxind - 1)) {
        change[id] <- getchange(tauhat, gofstat, id)
      }
    } else {
      EEl <- sim.g(y, tauhat[maxind - 1] + 1, tauhat[maxind + 1])
      rl <- gTests::g.tests(
        EEl,
        1:(tauhat[maxind] - tauhat[maxind - 1]),
        (tauhat[maxind] - tauhat[maxind - 1] + 1):(tauhat[maxind + 1] - tauhat[maxind - 1])
      )
      EEr <- sim.g(y, tauhat[maxind] + 1, tauhat[maxind + 2])
      rr <- gTests::g.tests(
        EEr,
        1:(tauhat[maxind + 1] - tauhat[maxind]),
        (tauhat[maxind + 1] - tauhat[maxind] + 1):(tauhat[maxind + 2] - tauhat[maxind])
      )
      gofstat[maxind - 1] <- eval(parse(text = paste(
        "rl$", argchar, "$test.statistic",
        sep = ""
      )))
      gofstat[maxind] <- eval(parse(text = paste(
        "rr$", argchar, "$test.statistic",
        sep = ""
      )))
      for (id in max(1, (maxind - 2)):min((maxind + 1), length(change))) {
        change[id] <- getchange(tauhat, gofstat, id)
      }
    }
    gofoverall[i + 1] <- sum(gofstat) - c * (length(tauhat) - 2) * log(n) - sum(log(diff(tauhat)))
    i <- i + 1
    # cat(c(i, gofoverall[i], max(tauhat[4:length(tauhat)] - tauhat[1:(length(tauhat) - 3)]), "\n"))
  }

  if (length(tauhat) >= (minCpNum + 3) && n < maxSeq) {
    maxind <- which.max(change)
    mergetau[i] <- tauhat[maxind + 1]
    gofstat <- gofstat[-maxind]
    tauhat <- tauhat[-(maxind + 1)]
    EE <- sim.g(y, 1, n)
    r <- gTests::g.tests(EE, 1:tauhat[2], (tauhat[2] + 1):n)
    gofstat <- eval(parse(text = paste(
      "r$", argchar, "$test.statistic",
      sep = ""
    )))
    gofoverall[i + 1] <- gofstat - c * log(n) - sum(log(diff(tauhat)))
  }
  ind <- which.max(gofoverall)
  tauhat <- tauhat[-c(1, length(tauhat))]
  if (ind != length(gofoverall)) {
    tauhat <- c(tauhat, mergetau[length(mergetau):ind])
  }
  if (detail == TRUE) {
    return(list(
      tauhat = tauhat,
      gofSeq = gofoverall,
      mergeSeq = mergetau
    ))
  } else {
    return(tauhat)
  }
}
