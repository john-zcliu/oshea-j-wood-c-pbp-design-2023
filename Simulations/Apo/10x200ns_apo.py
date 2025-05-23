import pathlib

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout


def get_protein_energy(simulation, prmtop):
    # Identify protein atoms (simple way: all atoms in standard residues)
    protein_atoms = [atom.index for atom in prmtop.topology.atoms() if atom.residue.name not in ['HOH', 'NA', 'CL']]
    
    # Use a group-based approach to extract energy for protein atoms only
    total_energy = 0 * kilojoule_per_mole
    for force_index in range(simulation.system.getNumForces()):
        force = simulation.system.getForce(force_index)
        # Create a mask for protein atoms
        state = simulation.context.getState(getEnergy=True, groups={1 << force_index})
        force_energy = state.getPotentialEnergy()
        # Filter energy contributions to protein atoms only
        if hasattr(force, 'getEnergyFunction') or hasattr(force, 'getParticles'):
            total_energy += force_energy
    
    # Not an exact protein-only energy — exact decomposition requires energy group splitting in custom forces
    # For proper protein-only energy, OpenMM has no direct support — needs custom forces or atom masking.
    return total_energy

def run_sim(top_path, coord_path, output_path, sim_time, sim_num):
    print("Loading amber files...")
    prmtop = AmberPrmtopFile(str(top_path))
    inpcrd = AmberInpcrdFile(str(coord_path))
    print("Loading amber files... Done.")

    print("Creating system...")
    system = prmtop.createSystem(
        nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds
    )
    system.addForce(MonteCarloBarostat(1 * bar, 300 * kelvin))
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtosecond)
    platform = Platform.getPlatformByName("CUDA")
    simulation = Simulation(prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    print("Creating system... Done.")


    # Minimise energy
    print("Minimising energy...")
    simulation.minimizeEnergy()
    print("Minimising energy... Done.")
    # Setup logging for NPT
    log_frequency = 100_000 # 2 fs * 100,000 = 0.2 ns. So one frame every 0.2 ns. 500 frames for a 100 ns simulation
    simulation.reporters.append(DCDReporter(
        str(output_path / f"npt_production_{sim_num:02d}.dcd"),log_frequency))
    simulation.reporters.append(
        StateDataReporter(
            str(output_path / f"npt_production_{sim_num:02d}.csv"),
            log_frequency,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            volume=True,
            speed=True,
            time=True,
        )
    )

    # NPT production run (with a barostat for constant pressure rather than volume)
    print("Running NPT production...")
    for ns_passed in range(1, sim_time + 1):
        simulation.step(500_000) # run simulation for 500,000 steps, 1ns
        if not (ns_passed % 5): # "not" occurs every 5ns because 5%5 = 0
            simulation.saveState(str(output_path / f"npt_production_{ns_passed}ns.xml")) #Save checkpoint data
            simulation.saveCheckpoint(str(output_path / f"npt_production_{ns_passed}ns.chk")) #Save checkpoint simulation state
        print(f"Completed {ns_passed}ns...")
    print("Running NPT production... Done.")
    return
    

if __name__ == '__main__':
    import os
    
    # change directory to where the script is
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    top_path = pathlib.Path("1anf_malremoved_t3p.parm7")
    coord_path = pathlib.Path("1anf_malremoved_t3p.rst7")
    sim_time = 200
    for i in range(1, 11):
        sim_num = i
        output_path = pathlib.Path(f"simulation_{i:02d}/") # gives the number "i" with a 0 in front if single digit
        output_path = pathlib.Path(f"testsimulation_{i:02d}/") # gives the number "i" with a 0 in front if single digit
        output_path.mkdir(exist_ok=True)
        print(f"Starting simulation {i}...")
        run_sim(top_path, coord_path, output_path, sim_time, sim_num)
        print(f"Completed simulation {i}.")
