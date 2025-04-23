"""
This script runs a 200ns NPT simulation of the 1ANF protein in a water box using OpenMM.
1. extract the protein
2. add/fix the protein by adding/removing hydrogen atoms
3. add ions to balance the charge
4. set an appropriate periodic box size
5. solvate the protein with water molecules
6. setup loggers to log the protein structures per 10fs (log only protein, without water or ions)
7. setup loggers to log the potential energy of the structure per 10fs in sync
8. run simulation

Required packages:
conda install -c conda-forge openmm ambertools mdtraj
pip install openmm-mdanalysis-reporter
"""

import subprocess
import os
from openmm.app import *
from openmm import *
from openmm.unit import *
from Bio import PDB
from mdareporter import MDAReporter
from sys import stdout

# === STATIC CONFIGURATION ===
PDB_FILE = "1anf.pdb"
PDB_PROCESSED_FILE = "1anf_without_ligand.pdb"
TOTAL_SIMULATIONS = 100
TEMPERATURE = 300 * kelvin
PRESSURE = 1 * atmosphere
TIMESTEP = 2 * femtoseconds
TOTAL_NS = 200
NSTEPS = int((TOTAL_NS * 1000 * picoseconds) / TIMESTEP)
REPORT_INTERVAL = int(10 * picoseconds / TIMESTEP)
BOX_PADDING = 1.0 * nanometers


class ProteinEnergyReporter(object):
    def __init__(self, file_path: str, reportInterval: int, full_system: System, protein_atoms: list[int]):
        self.file_path = file_path
        self._interval = reportInterval
        self._protein_atoms = protein_atoms
        self._protein_system, self._force_groups = self._build_protein_system(full_system, protein_atoms)
        self._out = open(self.file_path, "w")
        self._out.write("time_fs,bond,angle,torsion,nonbonded,one_four,total_potential,kinetic\n")

    def describeNextReport(self, simulation):
        return (self._interval, True, True, False, True)

    def report(self, simulation, state):
        time_fs = state.getTime().value_in_unit(femtoseconds)
        pos = state.getPositions(asNumpy=True)[self._protein_atoms]
        vel = state.getVelocities(asNumpy=True)[self._protein_atoms]

        integrator = VerletIntegrator(1 * femtoseconds)
        context = Context(self._protein_system, integrator)
        context.setPositions(pos)
        context.setVelocities(vel)

        # Energy terms by group
        bond = context.getState(getEnergy=True, groups={self._force_groups["bond"]}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        angle = context.getState(getEnergy=True, groups={self._force_groups["angle"]}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        torsion = context.getState(getEnergy=True, groups={self._force_groups["torsion"]}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        nonbonded = context.getState(getEnergy=True, groups={self._force_groups["nonbonded"]}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        one_four = context.getState(getEnergy=True, groups={self._force_groups["one_four"]}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        kinetic = context.getState(getEnergy=True).getKineticEnergy().value_in_unit(kilojoules_per_mole)

        total_pe = bond + angle + torsion + nonbonded + one_four

        self._out.write(f"{time_fs:.1f},{bond:.3f},{angle:.3f},{torsion:.3f},{nonbonded:.3f},{one_four:.3f},{total_pe:.3f},{kinetic:.3f}\n")
        self._out.flush()

        del context, integrator

    def _build_protein_system(self, full_system: System, protein_atoms: list[int]):
        new_system = System()
        force_groups = {}
        group_counter = 0
        atom_map = {old: new for new, old in enumerate(protein_atoms)}

        for idx in protein_atoms:
            new_system.addParticle(full_system.getParticleMass(idx))

        for force in full_system.getForces():
            if isinstance(force, HarmonicBondForce):
                new_force = HarmonicBondForce()
                for i in range(force.getNumBonds()):
                    p1, p2, length, k = force.getBondParameters(i)
                    if p1 in atom_map and p2 in atom_map:
                        new_force.addBond(atom_map[p1], atom_map[p2], length, k)
                new_force.setForceGroup(group_counter)
                new_system.addForce(new_force)
                force_groups["bond"] = group_counter
                group_counter += 1

            elif isinstance(force, HarmonicAngleForce):
                new_force = HarmonicAngleForce()
                for i in range(force.getNumAngles()):
                    p1, p2, p3, angle, k = force.getAngleParameters(i)
                    if all(p in atom_map for p in [p1, p2, p3]):
                        new_force.addAngle(atom_map[p1], atom_map[p2], atom_map[p3], angle, k)
                new_force.setForceGroup(group_counter)
                new_system.addForce(new_force)
                force_groups["angle"] = group_counter
                group_counter += 1

            elif isinstance(force, PeriodicTorsionForce):
                new_force = PeriodicTorsionForce()
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                    if all(p in atom_map for p in [p1, p2, p3, p4]):
                        new_force.addTorsion(atom_map[p1], atom_map[p2], atom_map[p3], atom_map[p4],
                                             periodicity, phase, k)
                new_force.setForceGroup(group_counter)
                new_system.addForce(new_force)
                force_groups["torsion"] = group_counter
                group_counter += 1

            elif isinstance(force, NonbondedForce):
                # Create two force objects: one for nonbonded, one for 1-4 exceptions
                nb_force = NonbondedForce()
                nb_force.setNonbondedMethod(NonbondedForce.NoCutoff)
                nb_force.setUseDispersionCorrection(False)

                one_four_force = NonbondedForce()
                one_four_force.setNonbondedMethod(NonbondedForce.NoCutoff)
                one_four_force.setUseDispersionCorrection(False)

                for i in protein_atoms:
                    charge, sigma, epsilon = force.getParticleParameters(i)
                    nb_force.addParticle(charge, sigma, epsilon)
                    one_four_force.addParticle(0.0, 1.0, 0.0)  # neutral for 1-4 container

                for i in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                    if p1 in atom_map and p2 in atom_map:
                        nb_force.addException(atom_map[p1], atom_map[p2], 0.0, 1.0, 0.0)  # skip these in nb
                        one_four_force.addException(atom_map[p1], atom_map[p2], chargeProd, sigma, epsilon)

                nb_force.setForceGroup(group_counter)
                new_system.addForce(nb_force)
                force_groups["nonbonded"] = group_counter
                group_counter += 1

                one_four_force.setForceGroup(group_counter)
                new_system.addForce(one_four_force)
                force_groups["one_four"] = group_counter
                group_counter += 1

        return new_system, force_groups

    def __del__(self):
        self._out.close()


def extract_protein_only(input_pdb_path, output_pdb_path, protein_chain_id='A'):
    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()

    structure = parser.get_structure("structure", input_pdb_path)

    class ProteinSelect(PDB.Select):
        def accept_chain(self, chain):
            return chain.id == protein_chain_id

        def accept_residue(self, residue):
            hetfield, resseq, icode = residue.id
            # Exclude heteroatoms (HETATM) and waters (HOH)
            return hetfield == ' ' and residue.get_resname() != 'HOH'

        def accept_atom(self, atom):
            return True

    # Modify chain ID of selected residues to 'A'
    for model in structure:
        for chain in model:
            if chain.id == protein_chain_id:
                chain.id = 'A'  # Ensure chain ID is set to 'A'

    io.set_structure(structure)
    io.save(output_pdb_path, select=ProteinSelect())

def prepare_system(protein_pdb: str, tleap_script_path="tleap.in") -> tuple[str, str, list[int]]:
    with open(tleap_script_path, "w") as f:
        f.write(f"""
source leaprc.protein.ff14SB
source leaprc.water.tip3p
mol = loadpdb {protein_pdb}
addions mol Na+ 0
addions mol Cl- 0
solvatebox mol TIP3PBOX 8.0
saveamberparm mol system.prmtop system.inpcrd
savepdb mol protein_solvated.pdb
quit
""")
    subprocess.run(["tleap", "-f", tleap_script_path], check=True)

    pdb = PDBFile("protein_solvated.pdb")
    protein_atoms = [atom.index for atom in pdb.topology.atoms() if atom.residue.chain.index == 0]
    return "system.prmtop", "system.inpcrd", protein_atoms

def setup_simulation(prmtop_file: str, inpcrd_file: str, protein_atoms: list[int], output_dir: str) -> Simulation:
    platform = Platform.getPlatformByName("CUDA")
    prmtop = AmberPrmtopFile(prmtop_file)
    inpcrd = AmberInpcrdFile(inpcrd_file)

    system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
    system.addForce(MonteCarloBarostat(PRESSURE, TEMPERATURE))

    integrator = LangevinMiddleIntegrator(TEMPERATURE, 1/picosecond, TIMESTEP)

    simulation = Simulation(prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(inpcrd.positions)

    print("Minimizing...")
    simulation.minimizeEnergy()
    print("Minimizing...Done")

    # Reporters
    # simulation.reporters.append(DCDReporter(os.path.join(output_dir, "protein_only.dcd"),
    #                                         REPORT_INTERVAL, enforcePeriodicBox=False,
                                            # append=False, atomSubset=protein_atoms))
    simulation.reporters.append(MDAReporter(os.path.join(output_dir, 'traj.dcd'), REPORT_INTERVAL, enforcePeriodicBox=False))

    simulation.reporters.append(
        StateDataReporter(os.path.join(output_dir, "simulation_stats.csv"), 
                          REPORT_INTERVAL, step=True, kineticEnergy=True,
                          potentialEnergy=True, temperature=True, progress=True,
                          remainingTime=True, speed=True, totalSteps=NSTEPS)
    )
    
    # simulation.reporters.append(EnergyReporter(os.path.join(output_dir, "energy.txt"), REPORT_INTERVAL))
    simulation.reporters.append(
        ProteinEnergyReporter(
            file_path=os.path.join(output_dir, "protein_energy.csv"),
            reportInterval=REPORT_INTERVAL,
            full_system=system,
            protein_atoms=protein_atoms
        )
    )

    return simulation

def run_simulation(simulation: Simulation):
    steps_per_ns = int((1 * nanoseconds) / TIMESTEP)
    for ns_passed in range(1, TOTAL_NS + 1):
        simulation.step(steps_per_ns)
        print(f"Completed {ns_passed}ns...")
    print("Simulation complete.")

def main():
    extract_protein_only(PDB_FILE, PDB_PROCESSED_FILE)
    prmtop, inpcrd, protein_atoms = prepare_system(PDB_PROCESSED_FILE)
    
    for i in range(TOTAL_SIMULATIONS):
        sim_id = f"simulations/sim_{i+1:03d}"
        print(f"\n=== Starting simulation {sim_id} ===")
        os.makedirs(sim_id, exist_ok=True)

        sim = setup_simulation(prmtop, inpcrd, protein_atoms, output_dir=sim_id)
        run_simulation(sim)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
