from tqdm import tqdm

import struct

def inverse_rsqrt(number):
	threehalfs = 1.5
	x2 = number * 0.5
	y = number

	# evil floating point bit level hacking
	i = struct.unpack('I', struct.pack('f', y))[0]
	i = 0x5f3759df - (i >> 1)
	y = struct.unpack('f', struct.pack('I', i))[0]

	# 1st iteration
	y = y * (threehalfs - (x2 * y * y))

	# 2nd iteration, this can be removed
	# y = y * (threehalfs - (x2 * y * y))
	result_bits = struct.unpack('I', struct.pack('f', y))[0]
	size = struct.calcsize('I')

	if result_bits < 0 or result_bits >= (1 << (size * 8)):
		raise ValueError('result_bits out of range')

	return struct.unpack('f', struct.pack('I', result_bits))[0]

f = 1.0

for i in tqdm(range(100)):
    for r in tqdm(range(1000000)):
        n = 1.001 ** (r*i)
        f = inverse_rsqrt(n)
        f = f + 1

