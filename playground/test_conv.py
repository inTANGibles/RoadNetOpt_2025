def conv(in_size, out_channel, kernel_size, strip):
    a = in_size[0]
    b = int((a - kernel_size) / strip) + 1
    return b, b, out_channel


def pool(in_size, n):
    a = in_size[0]
    b = int(a / n)
    return b, b, in_size[2]


data_size = (512, 512, 4)
data_size = conv(data_size, 64, 4, 2)
data_size = pool(data_size, 2)
data_size = conv(data_size, 64, 4, 2)
data_size = pool(data_size, 2)
data_size = conv(data_size, 64, 3, 1)
data_size = pool(data_size, 2)
print(data_size)
