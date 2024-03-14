using Logging
# @TODO: figure out why enabling this logging breaks the progress bar
# using LoggingExtras, Dates
# using Turing: AdvancedVI

# # set up logging
# const date_format = "yyyy-mm-dd HH:MM:SS"


# function transformer_logger(log)
#   if (log._module === AdvancedVI || parentmodule(log._module) === AdvancedVI)
#     return log
#   end
#   merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
# end

# function warn_filter(log_args)
#   if log_args.level === Logging.Warn
#     return false
#   end
#   return true
# end

# global_logger(TransformerLogger(transformer_logger, global_logger()))
# surpress_warnings = EarlyFilteredLogger(warn_filter, global_logger())