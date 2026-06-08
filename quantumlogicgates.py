import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import qutip
import cudaq
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid
from scipy.linalg import expm # Matrix exponential
import imageio
import os
from qiskit import QuantumCircuit
from scipy.linalg import sqrtm # This is to square root matrices

# First, we want to introduce the Pauli Matrices
matx = np.array([[0, 1], [1, 0]])
maty = np.array([[0, -1j], [1j, 0]])
matz = np.array([[1, 0], [0, -1]])

# We want to ask what values of ε, γ, ω should be used when visualising the oscillations.

ε = float(input("What value of ε (energy level spacing) do you want? ")) # If using Python Debugger in VSCode, the code may not work when the debugger is first ran due to this line.
γ = float(input("What value of γ (coupling strength) do you want? "))
ω = float(input("What value of ω (driving frequency) do you want? "))

# It is worth mentioning that the inequality sqrt((ε−ω)^2 +γ^2) ≪ ω must hold for oscillations and fast micro-motion to occur.

# What we want are 2 functions, the Hamiltonians, in their exact and RWA form

def hamil_exact(t, ε, γ, ω):
    """
    This is the exact time dependent Hamiltonian, suggested in the introductory lectures, and what will be used as the Hamiltonian when deriving logic gates.
    H(t) = (ε/2) * Z + γ * X * cos(ω * t)
    """
    return (ε / 2) * matz + γ * matx * np.cos(ω * t)

def hamil_RWA(ε, γ, ω):
    """
    This is the Rotating Wave Approximation suggested by Isidor Isaac Rabi in 1937. This is time independent and is another way of deriving logic gates.
    The derivation of this will be included separately.
    H_RWA = 0.5 * (ε - ω) * Z + (γ/2) * X
    """
    return 0.5 * (ε - ω) * matz + (γ/2) * matx

# We'll also define the Schrodinger Equation, as this will be useful for solving a later ODE.

def schrodinger_exact(t, ψ):
    """
    This will return the RHS of the Schrodinger equation.
    (d/dt) * |ψ⟩ = -i * H * |ψ⟩
    We must note that the reduced planck constant is 1 here such that it doesn't appear next to d/dt * |ψ⟩
    """
    ψ = ψ.astype(complex)
    return -1j * hamil_exact(t, ε, γ, ω) @ ψ # Note that @ is matrix multiplication.

def schrodinger_RWA(t, ψ):
    """
    This will return the RHS of the Schrodinger equation.
    (d/dt) * |ψ⟩ = -i * H * |ψ⟩
    We must note that the reduced planck constant is 1 here such that it doesn't appear next to d/dt * |ψ⟩
    It must be noted that this is time independent in the function, yet a time parameter is important as we later need to solve an ODE that involves time (Discrete Schrodinger Equation).
    """
    ψ = ψ.astype(complex)
    return -1j * hamil_RWA(ε, γ, ω) @ ψ

t_axis = np.linspace(0,10,10000)

# We must also initialise the starting state of |ψ⟩. We see the (1, 0)^T as heads or |0⟩ (bra-ket notation), so we also see (0, 1)^T as tails or |1⟩
ψ_0 = np.array([1,0], dtype=complex)

schrodinger_exact_solution = solve_ivp(
    schrodinger_exact,
    [0,10],
    ψ_0,
    t_eval = t_axis,
    rtol = 1e-10,
    atol = 1e-10,
)

schrodinger_RWA_solution = solve_ivp(
    schrodinger_RWA,
    [0,10],
    ψ_0,
    t_eval = t_axis,
    rtol = 1e-10,
    atol = 1e-10,
)

# One thing we must also notice is that we have to transform the RWA solution into its rotating frame.

def rotating_frame_adjustment(t):
    """
    This transforms the RWA solution such that it matches |ψ′⟩ = exp(iZωt/2)|ψ⟩.
    """
    return np.array([
        [np.exp(1j * ω * t / 2), 0],
        [0, np.exp(-1j * ω * t / 2)]
    ])

def realised_func(τ):
    """
    This function will be used to find our logic gates. It returns the realised sought gate function:
    U(τ) = exp(-i * H_RWA * τ)
    """
    return expm(-1j * hamil_RWA(ε, γ, ω) * τ) 

exact_prob0 = np.abs(schrodinger_exact_solution.y[0])**2
exact_prob1 = np.abs(schrodinger_exact_solution.y[1])**2
exact_absnorm = exact_prob0 + exact_prob1
exact_prob0 = exact_prob0 / exact_absnorm

RWA_prob0 = np.abs(schrodinger_RWA_solution.y[0])**2
RWA_prob1 = np.abs(schrodinger_RWA_solution.y[1])**2
RWA_absnorm = RWA_prob0 + RWA_prob1
RWA_prob0 = RWA_prob0 / RWA_absnorm

# The line above forces the probability to always stay between 0 and 1.

# Now we need to plot the probability against the time
plt.plot(schrodinger_exact_solution.t, exact_prob0, color='r')
plt.plot(schrodinger_RWA_solution.t, RWA_prob0, color='b')
plt.xlabel("Time: t")
plt.ylabel(r"$|\langle0|\psi(t)\rangle|^2$")
plt.xlim(0,10)
plt.ylim(0,1.05)
plt.legend(loc = 'upper right', labels=['Exact Hamiltonian', 'RWA Hamiltonian']) # Legend
plt.savefig("probability_plot.png")

hadamard = (1/np.sqrt(2))*(matx + matz)

def phase_gate(ϕ):
    """
    The phase gate requires a phi that ensures rotation. This can be seen as a linear transformation where |0⟩ -> |0⟩ and |1⟩ -> exp(iϕ) * |1⟩.
    """
    return np.array([[1, 0], [0, np.exp(1j * ϕ)]])

# These Hamiltonians are required for the Hadamard and Phase Gate. We seek this with our physical and RWA Hamiltonian from earlier.

if np.allclose(phase_gate(np.pi), matz): 
    print("The Z Pauli matrix and Phase gate with pi as the angle are equal.")
else:
    print("Something went wrong!")

def inner_product(f, g, frame_transform_f=False, frame_transform_g=False):
    """
    This function will compute the inner product of 2 functions, which helps us measure the overlap of the functions.
    It is often seen by the notation <f,g>. For our examples, the inner product will be defined on [0,T]. The T will be our time interval in which we solved the ODE earlier.
    If you want to use an exact Hamiltonian, you will want to set frame_transform to the function (set to True) in order for everything to be in a rotating frame.
    """

    fidelity = []
    for i in range(f.shape[1]):
        ψ_f = f[:, i] # This rotates the exact function into the rotating frame
        ψ_g = g[:, i]
        if frame_transform_f is True:
            rot_adj_f = rotating_frame_adjustment(t_axis[i])
            ψ_f = rot_adj_f @ ψ_f
            # This ensures we get the ψ' that we need
        if frame_transform_g is True:
            rot_adj_g = rotating_frame_adjustment(t_axis[i])
            ψ_g = rot_adj_g @ ψ_g
        overlap = np.vdot(ψ_f, ψ_g) # For vdot, check https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
        fidelity.append(np.abs(overlap)**2)
    fidelity = np.array(fidelity)
    average_fidelity = trapezoid(fidelity, t_axis) / (t_axis[-1] - t_axis[0]) # This computes an average
    fidelityerror = 1 - min(fidelity)
    return average_fidelity, fidelityerror

# Let's check how close the Exact and RWA Hamiltonians are.

print(inner_product(schrodinger_exact_solution.y,schrodinger_RWA_solution.y, True))
# We've now REASONABLY confirmed that the Exact Hamiltonian and RWA Hamiltonian, under a frame rotating at ω, are very similar to eachother graphically. This means we can use the RWA estimate. This result is found by testing tuples.
# Now we want to find the correct (ε, γ, ω) that achieves a Hadamard gate and Phase Gate.

ϕ_interval = np.linspace(0, 2*np.pi, 10000)
for ϕ in ϕ_interval:
    τ = (2 * ϕ) / (ε - ω)
    if np.allclose(phase_gate(ϕ), realised_func(τ)):
        print(f"Gate realised for ϕ={ϕ}")
    # This code doesn't actually work all that well, the matrix matches for ϕ=0, ϕ=2π, which doesn't help as this is where e^(iπ) = 1
    # This code however is quite useful. It has defined ϕ for later calculations.


# The global comparison below is more general, working for the Hadamard and Phase gate as it compares the matrices once the complex rotation is removed, which is physically irrelevant.
def global_phase_comparison(U, V, tol=1e-8):
    """
    This function takes in 2 matrices that you want to compare the properties of, disregards the global phase difference.
    """
    overlap = np.vdot(U.flatten(), V.flatten())
    if np.abs(overlap) < tol:
        return False
    phase = overlap / np.abs(overlap)
    return np.allclose(U, V / phase, atol=tol)

# We're also going to define a more powerful version of the Global Phase Comparison, that involves fidelity.
def gate_fidelity(U, V):
    """
    Say F is fidelity. Fidelity is the accuracy of quantum operations, saying how close "real quantum processes match the ideal, error-free processes. High fidelity means quantum gates and measurements are functioning correctly and producing reliable results, whereas low fidelity indicates more frequent errors that can corrupt computations". 
    See https://postquantum.com/quantum-computing/fidelity-quantum/#the-fidelity-imperative for more information.
    This returns a value on the interval [0,1]. If F > 0.99, the gate is realised correctly. 0.9 < F < 0.99, this is good but not excellent. F < 0.9 means the realisation hasn't been found for a chosen τ, γ.
    """
    return np.abs(np.trace(U.conj().T @ V)) / U.shape[0]

if np.isclose(ε, ω):
    print("Cannot realise the Phase Gate for ε = ω.")
else:
    if global_phase_comparison(phase_gate(ϕ), realised_func((2 * ϕ) / (ε - ω))): # We've picked τ = (2 * ϕ) / (ε - ω), γ = 0
        print("The Phase Gate has been realised for τ = (2 * ϕ) / (ε - ω), γ = 0.")
    else:
        print("The Phase Gate hasn't been realised due to a choice of τ and γ.")

@cudaq.kernel
def phase_kernel():
    qubit = cudaq.qubit() # This is a qubit initialised as |0⟩
    s(qubit) # S gate
    mz(qubit) # Quantum measurement
phase_result = cudaq.sample(phase_kernel, shots_count=10000)
print(phase_result)
# The result is {0:10000}, meaning we observe |0⟩ every single time.

@cudaq.kernel
def hsh_kernel():
    qubit = cudaq.qubit()
    h(qubit)
    s(qubit)
    h(qubit)
    mz(qubit)
hsh_result = cudaq.sample(hsh_kernel, shots_count=10000)
print(hsh_result)
# Now we see that |0⟩ is observed roughly 50% of the time and |1⟩ is observed roughly 50% of the time.

# We now want to look at the Hadamard gate. This has a very different τ and γ. We pick τ = π/(sqrt(2) * γ), resulting in U = iH, where H is Hadamard. γ = ε - ω. Using direct substitution, τ = π/(sqrt(2) * (ε - ω))
if np.isclose(ε, ω):
    print("Cannot realise the Hadamard Gate for ε = ω.")
else:
    if (global_phase_comparison(hadamard, realised_func(np.pi/(np.sqrt(2) * (ε - ω))))):
        print("The Hadamard Gate has been realised for τ = π/(sqrt(2) * γ), γ = ε - ω.")
        print(f"The gate fidelity is {gate_fidelity(hadamard, realised_func(np.pi/(np.sqrt(2) * (ε - ω))))}.")
    else:
        print("The Hadamard Gate hasn't been realised due to a choice of τ and γ.")
        print(f"The gate fidelity is {gate_fidelity(hadamard, realised_func(np.pi/(np.sqrt(2) * (ε - ω))))}.")

# Now we do a quick check with CUDA-Q to check the probabilities, we should roughly find {0:5000, 1:5000}.
@cudaq.kernel
def hadamard_kernel():
    qubit = cudaq.qubit()
    h(qubit)
    mz(qubit)
hadamard_result = cudaq.sample(hadamard_kernel, shots_count=10000)
print(hadamard_result)
# We obtained the correct result, roughly 50% are |0⟩ and the rest are |1⟩.

# We've obtained the 2 necessary 1 qubit gates indicated in the project brief. 
# We can now obtain a few more gates using the Hadamard and Phase gate.

phase_pi8 = phase_gate(np.pi/4) # Otherwise known as the T gate.
phase_s = phase_gate(np.pi/2)

if (global_phase_comparison(phase_gate(ϕ), expm(-1j * ϕ * matz))):
    print("The linear transformation form of the Phase Gate is equal to the sought form. This is due to a global phase difference that doesn't matter upon taking a measurement.")
    print(f"The gate fidelity is {gate_fidelity(phase_gate(ϕ), expm(-1j * ϕ * matz))}.")
else:
    print("There is difference between the definitions of the Phase Gate, meaning something is seriously wrong.")
    print(f"The gate fidelity is {gate_fidelity(phase_gate(ϕ), expm(-1j * ϕ * matz))}.")

# Now we actually have six 1-Qubit gates, the X (NOT), Y, Z (Phase, ϕ = π), Hadamard, Phase S (ϕ = π/2), Phase P (ϕ), Phase T (ϕ = π/4)  
# What if we want to visualise these gates with the Bloch sphere?
# Before we do this we have to now define the Quantum Object version of each Hamiltonian, which will be denoted by _QObj

def hamil_exact_QObj(t, ε, γ, ω):
    """
    This is the exact time dependent Hamiltonian, suggested in the introductory lectures, and what will be used as the Hamiltonian when deriving logic gates.
    H(t) = (ε/2) * Z + γ * X * cos(ω * t)
    This is in Quantum Object form, used for representation on the Bloch sphere with the QuTip package.
    """
    return (ε / 2) * qutip.sigmaz() + γ * qutip.sigmax() * np.cos(ω * t)

def hamil_RWA_QObj(ε, γ, ω):
    """
    This is the Rotating Wave Approximation suggested by Isidor Isaac Rabi in 1937. This is time independent and is another way of deriving logic gates.
    The derivation of this will be included separately.
    H_RWA = 0.5 * (ε - ω) * Z + (γ/2) * X
    This is in Quantum Object form, used for representation on the Bloch sphere with the QuTip package. 
    """
    return 0.5 * (ε - ω) * qutip.sigmaz() + (γ/2) * qutip.sigmax()

hadamard_QObj = (1/np.sqrt(2))*(qutip.sigmax() + qutip.sigmaz())

def realised_func_QObj(τ):
    """
    This function will be used to find our logic gates. It returns the realised sought gate function:
    U(τ) = exp(-i * H_RWA * τ)
    This is in Quantum Object form, used for representation on the Bloch sphere with the QuTip package.
    """
    τ = float(τ)
    return (-1j * hamil_RWA_QObj(ε, γ, ω) * τ).expm() # .expm() is used to make the matrix a QObj


"""
os.makedirs("frames", exist_ok=True)
times_phase = np.linspace(0, 2*np.pi, 100)
ϕ_bloch = times_phase
# It makes sense to start the phase gate at |+> instead of |0> as a Z rotation would not affect |0>
ψ_plus_QObj = (qutip.basis(2,0) + qutip.basis(2,1)).unit() # This applies the canonical basis of R^2, then makes it unitary (size is 1). This is used such that a Quantum Object is obtained
filenames_phase = []
for i, t in enumerate(times_phase):
    phase_bloch = realised_func_QObj((2 * ϕ_bloch) / (ε - ω)) # We must pick our (ε, γ, ω) tuple correctly. We also use time as the angle
    ψ_t_phase = (phase_bloch * ψ_plus_QObj).unit()
    x_phase = np.real(qutip.expect(qutip.sigmax(), ψ_t_phase))
    y_phase = np.real(qutip.expect(qutip.sigmay(), ψ_t_phase))
    z_phase = np.real(qutip.expect(qutip.sigmaz(), ψ_t_phase))
    fig_phase = plt.figure(figsize=(6,6))
    b_phase = qutip.Bloch(fig=fig_phase)
    b_phase.add_vectors([x_phase, y_phase, z_phase])
    xs_phase = []
    ys_phase = []
    zs_phase = []
    for t_prev in times_phase[:i+1]:
        phase_bloch_prev = realised_func_QObj((2 * t_prev) / (ε - ω))
        ψ_t_phase_prev = (phase_bloch_prev * ψ_plus_QObj)
        xs_phase.append(np.real(qutip.expect(qutip.sigmax(), ψ_t_phase_prev)))
        ys_phase.append(np.real(qutip.expect(qutip.sigmay(), ψ_t_phase_prev)))
        zs_phase.append(np.real(qutip.expect(qutip.sigmaz(), ψ_t_phase_prev)))
    b_phase.add_points([xs_phase, ys_phase, zs_phase], meth='m')
    b_phase.point_color = ['red']
    b_phase.vector_color = ['blue']
    b_phase.point_size = [8]
    b_phase.render()
    filename_phase = f"frames/frame_{i:03d}.png"
    fig_phase.savefig(filename_phase, dpi=150)
    filenames_phase.append(filename_phase)
    plt.close(fig_phase)
with imageio.get_writer("bloch_evolution_phase.gif", mode="I", duration=0.08) as writer:
    for filename_phase in filenames_phase:
        image = imageio.imread(filename_phase)
        writer.append_data(image)

"""
# We obtain an anti-clockwise rotation about the Z-axis, as expected from the realised gate. This is commented out as to not waste execution time on reproducing the same gif.
# NOw we want to animate the Hadamard gate time evolution. This may seem a bit more difficult at first as the Hadamard doesn't depend on time, but we can force this.
"""
ψ_0_QObj = qutip.basis(2,1)
times_hadamard = np.linspace(0, np.pi, 80)
filenames_hadamard = []
for i, t in enumerate(times_hadamard):
    hadamard_bloch = (-1j * hadamard_QObj * t).expm()
    ψ_t_hadamard = hadamard_bloch * ψ_0_QObj
    x_hadamard = np.real(qutip.expect(qutip.sigmax(), ψ_t_hadamard))
    y_hadamard = np.real(qutip.expect(qutip.sigmay(), ψ_t_hadamard))
    z_hadamard = np.real(qutip.expect(qutip.sigmaz(), ψ_t_hadamard))
    fig_hadamard = plt.figure(figsize=(6,6))
    b_hadamard = qutip.Bloch(fig=fig_hadamard)
    b_hadamard.add_vectors([x_hadamard, y_hadamard, z_hadamard])
    xs_hadamard = []
    ys_hadamard = []
    zs_hadamard = []
    for t_prev in times_hadamard[:i+1]:
        hadamard_bloch_prev = ((-1j * hadamard_QObj * t_prev).expm()) * ψ_0_QObj
        xs_hadamard.append(np.real(qutip.expect(qutip.sigmax(), hadamard_bloch_prev)))
        ys_hadamard.append(np.real(qutip.expect(qutip.sigmay(), hadamard_bloch_prev)))
        zs_hadamard.append(np.real(qutip.expect(qutip.sigmaz(), hadamard_bloch_prev)))
    b_hadamard.add_points([xs_hadamard, ys_hadamard, zs_hadamard], meth='m')
    b_hadamard.render()
    filename_hadamard = f"frame_{i:03d}.png"
    fig_hadamard.savefig(filename_hadamard, dpi=150)
    filenames_hadamard.append(filename_hadamard)
    plt.close(fig_hadamard)
with imageio.get_writer("bloch_evolution_hadamard_1state.gif", duration = 0.08) as writer:
    for f in filenames_hadamard:
        writer.append_data(imageio.imread(f))
"""

# We should get the application of 2 Hadamards, which goes |0⟩ H |+⟩ H |0⟩
# Similarly, we would see |1⟩ H |-⟩ H |1⟩

if np.isclose(γ, 0):
    print("Cannot realise the X gate for γ = 0.")
else:
    if global_phase_comparison(realised_func(np.pi/γ), matx):
        print("The NOT gate has been realised for τ = π/γ, γ ≠ 0 and ε = ω.")
        print(f"The gate fidelity is {gate_fidelity(realised_func(np.pi/γ), matx)}.")
    else:
        print("The NOT gate hasn't been realised due to a choice of τ and γ.")
        print(f"The gate fidelity is {gate_fidelity(realised_func(np.pi/γ), matx)}.")
# You must have γ = 2, τ = π/γ

# Now we explore a few more unfamiliar gates, the sqrt(X) gate, the Rotation Operator gates and the General Single Qubit Rotation. These are all 2x2 matrix logic gates.
# Before we start with the sqrt(X) gate, it is useful to know that the CUDA-Q kernel we used earlier with HSH gate simulation is actually equal to the sqrt(X) gate.

if np.isclose(γ, 0):
    print("Cannot realise the sqrt(X) gate for γ = 0.")
else:
    if global_phase_comparison(realised_func(np.pi/(2*γ)), sqrtm(matx)):
        print("The sqrt(X) gate has been realised for τ = π/2γ, γ ≠ 0 and ε = ω.")
        print(f"The gate fidelity is {gate_fidelity(realised_func(np.pi/(2*γ)), sqrtm(matx))}.")
    else:
        print("The sqrt(X) gate hasn't been realised due to a choice of τ and γ.")
        print(f"The gate fidelity is {gate_fidelity(realised_func(np.pi/(2*γ)), sqrtm(matx))}.")
# It must be noted that the sqrt(X) gate has only been tested for the trivial case that the coefficient of π is 1, instead of any other odd integer.
def rotation_X(θ):
    """
    This is the Rotating X gate, seen as R_X(θ) = exp(-0.5 * i * X * θ)
    """
    return expm(-0.5 * 1j * matx * θ)
def rotation_Y(θ):
    """
    This is the Rotating Y gate, seen as R_Y(θ) = exp(-0.5 * i * Y * θ)
    """
    return expm(-0.5 * 1j * maty * θ)
def rotation_Z(θ):
    """
    This is the Rotating Z gate, seen as R_Z(θ) = exp(-0.5 * i * Z * θ)
    """
    return expm(-0.5 * 1j * matz * θ)

θ_interval = np.linspace(0, 2*np.pi, 10000)
if np.isclose(γ, 0):
    print("Cannot realise the X Rotation gate for γ = 0.")
else:
    realised_all_rotX = all(
    global_phase_comparison(realised_func(θ / γ), rotation_X(θ))
    for θ in θ_interval
    )  
    if realised_all_rotX:
        print("The X Rotation Gate has been realised for τ = θ / γ, γ ≠ 0 and ε = ω.") # This is a weird looking if else statement. This was used in an attempt to fix an iteration across 10,000 values.
    else: 
        print("X Rotation Gate not realised for all θ.")


realised_all_rotZ = all(
    global_phase_comparison(realised_func(θ / (ε-ω)), rotation_Z(θ))
    for θ in θ_interval
    )  
if realised_all_rotZ:
    print("The Z Rotation Gate has been realised for τ = θ / (ε-ω) and γ = 0.")
else:
    print("Z Rotation Gate not realised for all θ.")
    
# Now we get to the Y Rotation Gate. This doesn't really work with our current H_RWA Hamiltonian. We actually have to use a unique identity.
realised_all_rotY = all(
    global_phase_comparison(phase_s @ rotation_X(θ) @ phase_s.conj().T, rotation_Y(θ))
    for θ in θ_interval
)
if realised_all_rotY:
    print("The Y Rotation Gate has been realised for τ = θ / γ, γ ≠ 0 and ε = ω. This was performed using S * R_x * S^†.")
else:
    print("Y Rotation Gate not realised for all θ.")

# We now want to check a different identity, H * R_Z * H = R_X
realised_all_identityX = all(
    global_phase_comparison(hadamard @ rotation_Z(θ) @ hadamard, rotation_X(θ)) # Check Nielsen-Chuang, Quantum Computation and Quantum Information, 10th Anniversary Edition, Pg 177
    for θ in θ_interval
)

if realised_all_identityX:
    print("We have found an identity for R_x that expresses it in terms of Hadamard and R_z.")
else:
    print("The identity doesn't hold.")

def gen_sin_q_rot(θ, ϕ, λ):
    """
    This is the General Single Qubit Rotation gate. It is a generalisation of rotations and can be represented as U(θ, ϕ, λ) = Rz​(ϕ)Ry​(θ)Rz​(λ).
    """
    return np.array([[np.cos(θ/2), -np.exp(1j * λ) * np.sin(θ/2)], [np.exp(1j * ϕ) * np.sin(θ/2), np.exp(1j * (ϕ + λ)) * np.cos(θ/2)]])

realised_gen_q_rot = all(
    global_phase_comparison(gen_sin_q_rot(θ, ϕ, λ), rotation_Z(ϕ) @ rotation_Y(θ) @ rotation_Z(λ))
    for θ, ϕ, λ in zip(
        np.random.uniform(0, 2*np.pi, 10000), # This produces random values on the interval [0, 2π], a total of 10000 times.
        np.random.uniform(0, 2*np.pi, 10000),
        np.random.uniform(0, 2*np.pi, 10000)
    )
)
if realised_gen_q_rot:
    print("We have found the General Single Qubit Rotation Gate, defined via R_z and R_y with the parameter tuple (θ, ϕ, λ).")
else:
    print("We haven't found the correct expression for the General Single Qubit Rotation Gate.")

# We've now obtained all 1-Qubit gates that are typically used. Now we want to consider how these gates can actually be used, for example, in Quantum Fourier Transforms.
# Note that we will have to use Quantum Objects to get use out of the basis function.

# This is all taken from Nielsen & Chuang Quantum Computation and Quantum Information 10th Anniversary Edition Page 217 Onwards
# This also uses https://nvidia.github.io/cuda-quantum/latest/examples/python/visualization.html

def discrete_fourier_transform(x):
    """
    This is the definition of a Discrete Fourier Transform, seen as y_k = 1/sqrt(N) * the sum from j = 0 to N-1 of x_j * exp((2π * i * j * k)/N)
    N = length of the vector x
    x is a vector containg x_0, ..., x_N-1 where all these values are complex numbers.
    """
    N = len(x)
    y = np.zeros(N, dtype=complex) # We defined y, the output vector as a list for now
    for k in range(N):
        sum = 0
        for p in range(N):
            sum += np.exp((2*np.pi * 1j * p * k) / N) * x[p]
        y[k] = sum / np.sqrt(N) 
    y_Qobj = qutip.Qobj(y)
    return y, y_Qobj

def quantum_discrete_fourier_transform(state):
    """
    This is the definition of a Quantum Fourier Transform, applied on a state. This is seen as |j⟩ -> 1/sqrt(N) * sum from k = 0 to N-1 of exp((2π * i * j * k)/N) * |k⟩
    N = length of the state vector
    state is the state vector, such as |0⟩ or |1⟩
    """
    N = len(state)
    output_state = np.zeros(N, dtype=complex) # We define ahead of time that the vector output will be complex type, length N
    for k in range(N):
        sum = 0
        for p in range(N):
            sum += state[p] * np.exp((2*np.pi * 1j * p * k) / N)
        output_state[k] = sum / np.sqrt(N)
    output_state_Qobj = qutip.Qobj(output_state)
    return output_state, output_state_Qobj

def qft_matrix(n_qubits):
    """
    This is an alternate definition of the Quantum Fourier Transform. It takes in a value n_qubits, and outputs the matrix that indicates a QFT for n qubits. If n = 2, we expect the Hadamard gate.
    """
    N = 2**n_qubits
    mat_output = np.zeros((N,N), dtype=complex)
    for k in range(N):
        for p in range(N):
            mat_output[k, p] = np.exp((2*np.pi * 1j * p * k) / N)
    mat_output = mat_output / np.sqrt(N)
    mat_output_Qobj = qutip.Qobj(mat_output)
    return mat_output, mat_output_Qobj

# Note that the Qobj form of each output is present in case we need to use any Quantum Python Package that renders NumPy incompatible for certain functions.
# Now we check our estimate for a 1-qubit QFT

if (global_phase_comparison(qft_matrix(1)[0], hadamard)):
    print("The matrix for a 1-qubit Quantum Fourier Transform is the same as a Hadamard gate.")
else:
    print("The matrix for a 1-qubit Quantum Fourier Transform is not the Hadamard gate, meaning something went wrong.")
print(qft_matrix(2)[0])

def dyadic_rational_phase_gate(k):
    """
    This is the Dyadic Rational Phase Gate is used for Quantum Fourier Transforms. We see this as R_k.
    This is also known as Controlled Phase Rotation. This can be seen in the cudaq.kernel where ctrl / cr1 etc. indicates the Dyadic Rotational Phase Gate.
    """
    return phase_gate(2*np.pi / 2**k)

# Now we want to create the 1, 2 and 3 Qubit Circuits for a QFT
@cudaq.kernel
def qft_1q():
    q = cudaq.qvector(1) # We start with |0⟩
    h(q[0])
    # mz(q)
# qft_1q_result = cudaq.sample(qft_1q, shots_count=10000)

# print(qft_1q_result) We have measured the probabilities of each state for each QFT. This results in roughly 1/2 probability for each state.
# It must be noted that these measurements are commented out, as the qft_matrix measurement against the expected outcome that we perform later will fail after a measurement, as the system collapses and every measurement reads one state.

print(cudaq.draw(qft_1q))
print(cudaq.draw('latex', qft_1q))
state_qft_1q = cudaq.get_state(qft_1q)
print(state_qft_1q)

# We can also use Qiskit to make png's of the Quantum circuits
qc_1 = QuantumCircuit(1) # This is a Quantum circuit with 1 Qubit. We start with |0⟩
qc_1.h(0) # Applies the QFT
qft_1q_fig = qc_1.draw(output='mpl')
qft_1q_fig.savefig("qft_1q.png", dpi=300)

@cudaq.kernel
def qft_2q():
    q = cudaq.qvector(2) # We start with |00⟩
    h(q[0])
    cr1(np.pi/2, q[1], q[0]) # This is the same as dyadic_rational_phase_gate(2)
    h(q[1])
    swap(q[0], q[1])
    # mz(q)
# qft_2q_result = cudaq.sample(qft_2q, shots_count=10000)
# print(qft_2q_result) This results in roughly 1/4 probability for each state.

print(cudaq.draw(qft_2q))
print(cudaq.draw('latex', qft_2q))
state_qft_2q = cudaq.get_state(qft_2q)
print(state_qft_2q)
state_qft_2q_numpy = np.array(state_qft_2q)

# Now we add the Qiskit plot
qc_2 = QuantumCircuit(2) # We start with |00⟩
qc_2.h(0)
qc_2.cp(np.pi/2, 1, 0) # This is the same as dyadic_rational_phase_gate(2)
qc_2.h(1)
qc_2.swap(0,1)
qft_2q_fig = qc_2.draw(output='mpl')
qft_2q_fig.savefig("qft_2q.png", dpi=300)

# We also want to make a quick confirmation
if global_phase_comparison(qft_matrix(2)[0] @ np.array(([1,0,0,0]), dtype=np.complex128), state_qft_2q_numpy):
    print("Our qft_matrix agrees with the expected outcome of the QFT on a 2 Qubit System.")
else:
    print("Something went wrong in the calculations.")

@cudaq.kernel
def qft_3q():
    q = cudaq.qvector(3) # We start with |000⟩
    h(q[0])
    cr1(np.pi / 2, q[1], q[0])
    cr1(np.pi / 4, q[2], q[0]) # This is the same as dyadic_rational_phase_gate(3)
    h(q[1])
    cr1(np.pi / 2, q[2], q[1])
    h(q[2])
    swap(q[0], q[2])
    # mz(q)
# qft_3q_result = cudaq.sample(qft_3q, shots_count=10000)
# print(qft_3q_result) This approximates to roughly 1/8 probability per state.

# Some samples taken are: { 000:1234 001:1265 010:1247 011:1276 100:1267 101:1241 110:1298 111:1172 }, { 000:1292 001:1306 010:1281 011:1222 100:1268 101:1188 110:1201 111:1242 }.
# If we iteratively find n samples for n large and average the number of samples for each state, you should roughly get 1250 for each state, confirming our expected result with high confidence.

print(cudaq.draw(qft_3q))
print(cudaq.draw('latex', qft_3q))
state_qft_3q = cudaq.get_state(qft_3q)
print(state_qft_3q)
state_qft_3q_numpy = np.array(state_qft_3q)

qc_3 = QuantumCircuit(3) # We start with |000⟩
qc_3.h(0)
qc_3.cp(np.pi/2, 1, 0)
qc_3.cp(np.pi/4, 2, 0) # This is the same as dyadic_rational_phase_gate(3)
qc_3.h(1)
qc_3.cp(np.pi/2, 2, 1)
qc_3.h(2)
qc_3.swap(0,2)
qft_3q_fig = qc_3.draw(output='mpl')
qft_3q_fig.savefig("qft_3q.png", dpi=300)

if global_phase_comparison(qft_matrix(3)[0] @ np.array(([1,0,0,0,0,0,0,0]), dtype=np.complex128), state_qft_3q_numpy):
    print("Our qft_matrix agrees with the expected outcome of the QFT on a 3 Qubit System.")
else:
    print("Something went wrong in the calculations.")

# We have now implemented the 1, 2 and 3 Qubit QFT's.

# We know the 3-Qubit State to be true via Nielsen & Chuang Quantum Computation and Quantum Information 10th Anniversary Edition Page 220

