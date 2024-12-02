def conv(in_size, out_channel, kernel_size, strip):
    a = in_size[0]
    b = int((a - kernel_size) / strip) + 1
    return b, b, out_channel


def pool(in_size, n):
    a = in_size[0]
    b = int(a / n)
    return b, b, in_size[2]



