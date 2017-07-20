import numpy as np  # generic math functions
import qutip as qt  #qutip for quantum states
from numpy.linalg import qr #QR decomposition

"""
This is an example code to generate a left-canonical MPS. Theoretical background can be found in Schollwocks
"The density-matrix renormalization group in the age of matrix product states".
This code is after his chapter 4.1.3. Since we aim for efficient computation we do the decomposition
QR and not with SVD. (See page 20)
The code is supposed to be simple and should give a first instruction how to handle the contractions and the
reshaping. It does not generalize at all and does not even work for N!=4.
""" 

def random_state(N): #This function produces a random state, which could be used later
    real = np.random.normal(0, 1, 2**N)             #makes random matrices for real and imaginary part
    imag = 1j*np.random.normal(0, 1, 2**N)
    psi = real + imag
    return qt.Qobj(psi, dims=[[2**N], [1]]).unit()  #generate quantum object, that has qubit tensor structure and normalize it
    
N= 4  # number of sites
d = 2 #physical dimension

#In this example we do not truncate anything. So it is not feasable for big N

M = [] #generate empty list, where we will put our matrices

#-------------------------------------------------------------------------------------------------------------
state = 1/np.sqrt(2)*(qt.tensor([qt.qstate('d')]*N) + qt.tensor([qt.qstate('u')]*N)) #generate simple state
state = state.full() #make numpy.array out of quantum object
#-------------------------------------------------------------------------------------------------------------

"""
We have to stick to a convention, which index of the tensor represents the physical index sigma.
This decision is arbitrary. In our case it will be the 2nd index of the tensor e.g. (a_i, sigma, a_j)
"""

#first we reshape the tensor c (Schollwok calls it a vector) into the matrix Psi
psi = state.reshape(d, d**(N-1))    #the indices of Psi are in the first step (sigma1, (sigma2 sigma3 ...) )
Q, R = qr(psi)                      #apply QR decomposition
M.append(Q.reshape(1,d,d))          #The 1st index is a dummy index (see eq. (42)), 2nd is sigma_1, 3rd is a_1
                                    #This reshaped Q is already our first matrix A of the MPS
psi = R.reshape(Q.shape[1]*d,d**(N-2))  #reshape R to the 2nd Psi in Eq. 42 (Psi_(a1 sigma2),(sigma3 ... sigmaL))
Q, R = qr(psi)                          #apply next QR
M.append(Q.reshape(d,d,d**2))           #again reshape Q, which is the 2nd A matrix
psi = R.reshape(Q.shape[1]*d,d**(N-3))  #reshape R to (Psi_(a2 sigma3),(sigma4))
Q, R = qr(psi)                          #next QR
M.append(Q.reshape(d**2,d,d))           #again reshape Q, which is the 3rd A matrix
psi = R.reshape(Q.shape[1]*d,d**(N-4))  #reshape R to (Psi_(a3 sigma4),1)
Q, R = qr(psi)                          #QR
M.append(Q.reshape(d,d,1))              #again reshape Q, which is the 4th A matrix

"""
We are done now. And have all the matrices for N=4. To show, that we actually have the MPS of the state above
we have to perform the sum from eq. 37.
Since in this very simple example we only have the basis states |0000> and |1111>. So only the multiplications
of A with the same physical index should lead to a result !=0.
"""
a = 1.0
b = 1.0
c = 1.0
for i in range(0,N):
    a = np.dot(a, M[i][:,0,:])  #a gives the result of the matrix multiplikation with indices sigma=0
    b = np.dot(b, M[i][:,1,:])  #b gives the result of the matrix multiplikation with indices sigma=1
    j = np.random.randint(2)    
    print(j)
    c = np.dot(c, M[i][:,j,:])  #c gives the result of the matrix multiplikation with random indices sigma={0,1}
                                #therefore c is only != if all indoices are equal (e.g. 0000 or 1111)
    
print(a, b, c)



