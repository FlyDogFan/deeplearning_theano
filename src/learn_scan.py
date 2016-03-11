import theano
import theano.tensor as T
import numpy

k = T.iscalar("k")
A = T.vector("A")

def step(result, a):
    result = a + result
    a = result + 1
    return result, a

[result, a], updates = theano.scan(fn=step,
                              outputs_info=[T.ones_like(A), A],
                              n_steps=k)

final_result = result
final_a = a

power = theano.function(inputs=[A,k], outputs=(final_result, final_a), updates=updates)

print(power(range(10),2))
print(power(range(10),4))