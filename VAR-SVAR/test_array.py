from numpy.f2py.crackfortran import beginpattern90


def test():
    a = [1, 2, 3]
    b = [ "a", "b", "c"]
    c = [[1 2 3]]
    print("A: ", a)
    print("A[1]", a[1])
    print("B:", b)
    print("B[3]", b[3])
