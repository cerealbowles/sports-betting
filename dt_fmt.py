"""Cross-platform strftime for Unix-style no-pad flags (%-d, %-I, etc.).

Windows' C runtime doesn't support the %-x spelling — it raises
ValueError: Invalid format string — and spells the same thing %#x instead.
fmt() tries the Unix flag first, falls back to the Windows flag, then
falls back to the plain zero-padded flag, so formatting never raises
regardless of host OS.
"""


def fmt(dt, fmt_str):
  try:
    return dt.strftime(fmt_str)
  except ValueError:
    try:
      return dt.strftime(fmt_str.replace('%-', '%#'))
    except ValueError:
      return dt.strftime(fmt_str.replace('%-', '%'))
