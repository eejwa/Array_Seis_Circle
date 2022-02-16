
def myround(x, prec=2, base=0.05):
    """
    Rounds the number 'x' to the nearest 'base' with precision 'prec'

    Parameters
    ----------
    x : float
        Number to be rounded.
    prec : int
        Number of decimal places for the rounded number.
    base : float
        The interval to be rounded nearest to.

    Returns
    -------
    The input number rounded to the nearest 'base' value.
    """
    return round(base * round(float(x) / base), prec)

def clip_traces(stream):
    """
    The traces in the stream may be of different length which isnt great for stacking etc.
    This function will trim the traces to the same length on the smallest trace size.

    Parameters
    ----------
    stream : Obspy stream object
        Any stream object which have traces with data in them.

    Returns
    -------
    stream : Obspy stream
        Data of equal length in time.
    """
    import numpy as np

    stimes = []
    etimes = []

    for trace in stream:
        stimes.append(trace.stats.starttime)
        etimes.append(trace.stats.endtime)

    stream = stream.trim(starttime=np.max(stimes), endtime=np.amin(etimes))

    return stream
