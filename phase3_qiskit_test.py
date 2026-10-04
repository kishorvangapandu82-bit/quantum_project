"""
============================================================
PHASE 3 — QISKIT BASIC TEST
NIFTY50-VQC Project

Purpose:
    Verify that Qiskit 2.5.2 + Qiskit-Aer 0.17.2 are installed
    correctly and working as expected on this machine.

    Demonstrates:
    1. Creating a quantum circuit with 2 qubits
    2. Applying a Hadamard gate (superposition)
    3. Applying a CNOT gate (entanglement)
    4. Measuring qubits
    5. Simulating with AerSimulator (1024 shots)
    6. Interpreting results

Circuit: Bell State |Φ+⟩ — the "Hello World" of quantum computing.

Expected result:
    ~50% measurements = '00'
    ~50% measurements = '11'
    Never '01' or '10'

    This is the signature of quantum entanglement.

Author: NIFTY50-VQC Project
Date:   2026-10-02
============================================================
"""

import sys

print("=" * 60)
print("PHASE 3 — QISKIT BASIC QUANTUM TEST")
print("=" * 60)

# ----------------------------------------------------------
# STEP 1: Check Python and package versions
# ----------------------------------------------------------
print("\n[STEP 1] Checking package versions...")
print(f"  Python      : {sys.version.split()[0]}")

import qiskit
print(f"  Qiskit      : {qiskit.__version__}")

import qiskit_aer
print(f"  Qiskit-Aer  : {qiskit_aer.__version__}")

# ----------------------------------------------------------
# STEP 2: Create the AerSimulator backend
# ----------------------------------------------------------
print("\n[STEP 2] Creating AerSimulator backend...")
from qiskit_aer import AerSimulator

simulator = AerSimulator(method='statevector')
print(f"  Backend name   : {simulator.name}")
print(f"  Simulation method : statevector")
print(f"  Status         : Ready")

# ----------------------------------------------------------
# STEP 3: Build a 2-qubit Bell State quantum circuit
# ----------------------------------------------------------
print("\n[STEP 3] Building a 2-qubit Bell State circuit...")
print("""
  What is a Bell State?
  ---------------------
  It is a 2-qubit quantum circuit that:
  1. Starts both qubits in state |0>  (like 2 coins both heads-up)
  2. Applies Hadamard to qubit 0      (coin 0 starts spinning)
  3. Applies CNOT(0 -> 1)             (coin 1 copies coin 0's spin)

  Result: Both qubits are now ENTANGLED.
  When measured, they always give the SAME result:
    both 0  OR  both 1  — never different!
""")

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Create a quantum register with 2 qubits
qr = QuantumRegister(2, name='q')

# Create a classical register with 2 bits (to store measurement results)
cr = ClassicalRegister(2, name='c')

# Create the circuit
circuit = QuantumCircuit(qr, cr)

# ----- Gate 1: Hadamard on qubit 0 -----
# H gate puts qubit 0 into superposition: |0> -> (|0> + |1>) / sqrt(2)
# Think of it as: the coin is now spinning (50% heads, 50% tails)
circuit.h(qr[0])

# ----- Gate 2: CNOT (Controlled-NOT) -----
# Control qubit: qr[0]   Target qubit: qr[1]
# If qr[0] is |1>, flip qr[1]. If qr[0] is |0>, do nothing.
# Since qr[0] is in superposition, this ENTANGLES both qubits.
circuit.cx(qr[0], qr[1])

# ----- Measurement -----
# Measure both qubits into classical bits
# This "collapses" the quantum superposition to a definite 0 or 1
circuit.measure(qr, cr)

print("  Circuit built successfully!")

# ----------------------------------------------------------
# STEP 4: Draw the circuit (text format)
# ----------------------------------------------------------
print("\n[STEP 4] Drawing the circuit (text format):")
print("-" * 40)
print(circuit.draw(output='text'))
print("-" * 40)
print("""
  Reading the circuit diagram:
  ----------------------------
  q_0: ─[H]─●─[M]   H = Hadamard gate (superposition)
             │        ● = Control qubit of CNOT
  q_1: ─────X─[M]   X = Target qubit of CNOT (flip if control=1)
                     M = Measurement
""")

# ----------------------------------------------------------
# STEP 5: Simulate the circuit
# ----------------------------------------------------------
print("\n[STEP 5] Simulating circuit (1024 shots)...")
print("""
  What is a 'shot'?
  -----------------
  Each 'shot' = one run of the quantum circuit.
  The circuit starts fresh, runs, and we record one measurement.

  We run 1024 shots to see the STATISTICAL distribution.
  (A real quantum computer runs the circuit many times because
   measurement is probabilistic — each run can give a different result.)
""")

from qiskit import transpile

# Transpile the circuit for the AerSimulator
# (This optimises the circuit for the specific backend)
compiled_circuit = transpile(circuit, simulator)

# Run the simulation
job = simulator.run(compiled_circuit, shots=1024)

# Get results
result = job.result()
counts = result.get_counts(compiled_circuit)

print(f"  Simulation complete!")
print(f"  Total shots run : 1024")
print(f"  Raw counts      : {counts}")

# ----------------------------------------------------------
# STEP 6: Interpret the results
# ----------------------------------------------------------
print("\n[STEP 6] Interpreting results...")
print("-" * 40)

total_shots = sum(counts.values())
print(f"\n  {'Outcome':<12} {'Count':<10} {'Percentage':<12} {'Meaning'}")
print(f"  {'-'*60}")

for outcome in sorted(counts.keys()):
    count = counts[outcome]
    pct = (count / total_shots) * 100
    if outcome == '00':
        meaning = "Both qubits = 0 (correct!)"
    elif outcome == '11':
        meaning = "Both qubits = 1 (correct!)"
    elif outcome == '01':
        meaning = "WARNING: unexpected!"
    elif outcome == '10':
        meaning = "WARNING: unexpected!"
    else:
        meaning = "Unknown"
    print(f"  {outcome:<12} {count:<10} {pct:<11.1f}%  {meaning}")

print(f"\n  Total: {total_shots} shots")

# ----------------------------------------------------------
# STEP 7: Verification
# ----------------------------------------------------------
print("\n[STEP 7] Verifying results are scientifically correct...")
print("-" * 40)

# A correct Bell State should ONLY produce '00' and '11'
unexpected = [k for k in counts.keys() if k not in ('00', '11')]

if unexpected:
    print(f"\n  RESULT: FAILED")
    print(f"  Unexpected outcomes detected: {unexpected}")
    print(f"  This should not happen in a Bell State.")
else:
    count_00 = counts.get('00', 0)
    count_11 = counts.get('11', 0)
    pct_00 = (count_00 / total_shots) * 100
    pct_11 = (count_11 / total_shots) * 100

    print(f"\n  Only '00' and '11' observed: CORRECT")
    print(f"  '00' : {count_00:4d} shots ({pct_00:.1f}%)")
    print(f"  '11' : {count_11:4d} shots ({pct_11:.1f}%)")
    print(f"  '01' :    0 shots ( 0.0%) — never observed")
    print(f"  '10' :    0 shots ( 0.0%) — never observed")

    # Check roughly 50/50 split (allow +/- 10% tolerance)
    balance_ok = abs(pct_00 - 50) < 10 and abs(pct_11 - 50) < 10

    if balance_ok:
        print(f"\n  Distribution is approximately 50/50: CORRECT")
        print(f"\n  *** VERIFICATION: PASSED ***")
        print(f"\n  This confirms:")
        print(f"  [1] Qiskit {qiskit.__version__} is working correctly")
        print(f"  [2] Qiskit-Aer {qiskit_aer.__version__} simulator is working")
        print(f"  [3] Hadamard gate creates superposition correctly")
        print(f"  [4] CNOT gate creates entanglement correctly")
        print(f"  [5] Measurement collapses quantum state correctly")
        print(f"  [6] AerSimulator statevector method is operational")
    else:
        print(f"\n  WARNING: Distribution is not close to 50/50.")
        print(f"  This is unusual. May need investigation.")

print("\n" + "=" * 60)
print("PHASE 3 — QISKIT BASIC TEST COMPLETE")
print("=" * 60)
