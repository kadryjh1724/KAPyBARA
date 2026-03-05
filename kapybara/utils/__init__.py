"""Utility sub-package.

Exports data-type conversion helpers, a timing decorator, a CSV trimmer,
Rich console helpers for notifications/warnings/errors, and the internal
validation error class.
"""

from kapybara.utils.convert import npy2Cint, npy2Cdouble, str2npy, npy2str, strfdelta2
from kapybara.utils.decorate import measureTime
from kapybara.utils.trim import trim_csv
from kapybara.utils.errors import _ValidationError
from kapybara.utils.cstring import prettyNotification, prettyWarning, prettyError
