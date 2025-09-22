# run_gsegmulti.R

# if (!require("jsonlite")) install.packages("jsonlite", dependencies = TRUE)
library(jsonlite)

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- dirname(sub(file_arg, "", args_full[grep(file_arg, args_full)]))

r_dir <- file.path(script_path, "gSegMulti", "R")
r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
suppressMessages(sapply(r_files, function(f) suppressWarnings(source(f))))

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]
alpha <- as.numeric(args[3])  # alpha
search_type <- args[4]  # search.type

# 데이터 로딩
X <- tryCatch({
  as.matrix(read.csv(input_file, header = FALSE))
}, error = function(e) {
  message("❌ 데이터 로딩 실패: ", conditionMessage(e))
  matrix(NA, nrow = 1, ncol = 1)
})

# gSegMulti 실행 (출력 숨기고 예외 방지)
res <- tryCatch({
  suppressMessages(
    suppressWarnings({
      tmp <- NULL
      capture.output({
        tmp <- gSegMulti(X, search.type = search_type, alpha = alpha)
      })
      tmp
    })
  )
}, error = function(e) {
  message("❌ gSegMulti 실행 실패: ", conditionMessage(e))
  NULL
})

# 결과를 항상 list(tauhat = ...) 형태로 변환
result <- tryCatch({
  if (is.list(res) && !is.null(res$tauhat)){
    if (!("tauhat" %in% names(res))) {
      list(tauhat = res)
    } else {
      list(tauhat = res$tauhat)
    }
  } else {
    list(tauhat = res)
  }
}, error = function(e) {
  message("❌ 결과 가공 실패: ", conditionMessage(e))
  list(tauhat = NULL)
})

# JSON 저장
tryCatch({
  write(toJSON(result, auto_unbox = TRUE, digits = 10), file = output_file)
}, error = function(e) {
  message("❌ JSON 저장 실패: ", conditionMessage(e))
  write('{"tauhat": null}', file = output_file)
})
