# =============================================================================
# export_session_info.R
# Captures the R version and the exact version of every package used in this
# project, writing R_session_info.txt next to this script. Run once after
# installing the packages so the R environment is documented for reproducibility.
# =============================================================================

here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile, mustWork = NA)),
                 error = function(e) NA_character_)
if (is.na(here) || here == "") here <- getwd()

pkgs <- c("caret", "dplyr", "ggplot2", "glmnet", "rms", "Hmisc", "Boruta",
          "naniar", "survival", "cmprsk", "readr", "tidyr")

lines <- c(paste("R version:", R.version.string),
           paste("Platform:", R.version$platform),
           paste("Recorded:", format(Sys.time())),
           "", "Package versions:",
           paste0("  ", pkgs, " ",
                  vapply(pkgs, function(p)
                      if (requireNamespace(p, quietly = TRUE))
                          as.character(utils::packageVersion(p)) else "NOT INSTALLED",
                      character(1))))

writeLines(lines, file.path(here, "R_session_info.txt"))
cat(paste(lines, collapse = "\n"), "\n")
cat("\nSaved:", file.path(here, "R_session_info.txt"), "\n")
