def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False